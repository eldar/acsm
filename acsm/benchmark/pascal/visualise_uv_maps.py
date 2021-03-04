from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import startup

import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pdb
from acsm.utils import visdom_render
from acsm.utils import transformations
from acsm.utils import visutil
from acsm.utils import mesh
from acsm.nnutils import uv_to_3d
from acsm.nnutils import misc as misc_utils
from acsm.utils.visualizer import Visualizer
from acsm.utils import bird_vis
from acsm.nnutils.nmr import NeuralRenderer
from acsm.utils import render_utils
from acsm.nnutils import icn_net, model_utils
from acsm.data import objects as objects_data
from acsm.nnutils import test_utils
"""
Script for testing on CUB.
Sample usage: python -m dcsm.benchmark.pascal.kp_transfer --split val --name <model_name> --num_train_epoch <model_epoch>
"""

from acsm.benchmark import pck_eval
from absl import app
from absl import flags
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import scipy.io as sio
import cPickle as pkl
import scipy.misc
import imageio

cm = plt.get_cmap('jet')
# from matplotlib import set_cmap
flags.DEFINE_boolean('visualize', False, 'if true visualizes things')
flags.DEFINE_integer('seed', 0, 'seed for randomness')
flags.DEFINE_boolean('mask_dump', False, 'dump mask predictions')
flags.DEFINE_string('mask_predictions_path', None, 'Mask predictions to load')
flags.DEFINE_boolean('never_deform', False, 'whether to use part deformations,')

opts = flags.FLAGS
# color_map = cm.jet(0)
kp_eval_thresholds = [0.05, 0.1, 0.2]


class UVMapTester(test_utils.Tester):
    def define_model(self, ):

        opts = self.opts
        self.img_size = opts.img_size
        self.offset_z = 5.0

        if opts.never_deform:
            init_stuff = {
                'mean_shape': self.mean_shape,
                'cam_location': self.cam_location,
                'offset_z': self.offset_z,
                'uv_sampler': self.uv_sampler,
                'kp_perm': self.kp_perm,
            }
        else:
            init_stuff = {
                'alpha': self.mean_shape['alpha'],
                'active_parts': self.part_active_state,
                'part_axis': self.part_axis_init,
                'kp_perm': self.kp_perm,
                'part_perm': self.part_perm,
                'mean_shape': self.mean_shape,
                'cam_location': self.cam_location,
                'offset_z': self.offset_z,
                'kp_vertex_ids': self.kp_vertex_ids,
                'uv_sampler': self.uv_sampler,
            }

        is_dataparallel_model = self.dataparallel_model(
            'pred', self.opts.num_train_epoch
        )
        self.model = icn_net.ICPNet(opts, init_stuff)
        if is_dataparallel_model:
            self.model = torch.nn.DataParallel(self.model)
        self.load_network(
            self.model,
            'pred',
            self.opts.num_train_epoch,
        )
        self.upsample_img_size = (
            (opts.img_size // 64) * (2**6), (opts.img_size // 64) * (2**6)
        )

        self.grid = misc_utils.get_img_grid(
            self.upsample_img_size
        ).repeat(opts.batch_size * 2, 1, 1, 1).to(self.device)

        self.offset_z = 5.0
        self.model.to(self.device)

        self.uv2points = uv_to_3d.UVTo3D(self.mean_shape)

        self.kp_names = self.dataloader.dataset.kp_names

        self.iter_count = 0

    def init_render(self, ):
        opts = self.opts
        if self.opts.never_deform:
            nkps = 0
        else:
            nkps = len(self.kp_vertex_ids)
        self.keypoint_cmap = [cm(i * 255 // nkps) for i in range(nkps)]
        model_obj_dir = osp.join(self.save_dir, 'model')
        visutil.mkdir(model_obj_dir)
        self.model_obj_path = osp.join(
            model_obj_dir, 'mean_{}.obj'.format(opts.pascal_class)
        )

        faces_np = self.mean_shape['faces'].data.cpu().numpy()
        verts_np = self.mean_shape['sphere_verts'].data.cpu().numpy()
        uv_sampler = mesh.compute_uvsampler(
            verts_np, faces_np, tex_size=opts.tex_size
        )
        uv_sampler = torch.from_numpy(uv_sampler).float().cuda()
        self.uv_sampler = uv_sampler.view(
            -1, len(faces_np), opts.tex_size * opts.tex_size, 2
        )
        self.verts_uv = self.mean_shape['uv_verts']
        self.verts_obj = self.mean_shape['verts']

        self.sphere_uv_img = scipy.misc.imread(
            osp.join(opts.cachedir, 'color_maps', 'sphere.png')
        )
        self.sphere_uv_img = torch.FloatTensor(self.sphere_uv_img) / 255
        self.sphere_uv_img = self.sphere_uv_img.permute(2, 0, 1)

        self.verts_obj = self.mean_shape['verts']

        if self.opts.never_deform:
            nkps = 0
        else:
            nkps = len(self.kp_vertex_ids)
        self.keypoint_cmap = [cm(i * 255 // nkps) for i in range(nkps)]

        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        self.renderer = visdom_render.RendererWrapper(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap,
            self.opts
        )

        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        renderer_no_light = visdom_render.RendererWrapper(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap,
            self.opts
        )
        renderer_no_light.vis_rend.set_light_status(False)
        renderer_no_light.vis_rend.set_bgcolor((255, 255, 255))
        self.renderer_no_light = renderer_no_light

    def init_dataset(self, ):
        opts = self.opts
        if opts.category == 'bird':
            self.dataloader = objects_data.cub_data_loader(opts, )
        elif opts.category in ['horse', 'sheep', 'cow']:
            self.dataloader = objects_data.imnet_pascal_quad_data_loader(
                opts, pascal_only=True
            )
        elif opts.category == 'car':
            self.dataloader = objects_data.p3d_data_loader(
                opts,
            )
            self.dataloader = objects_data.p3d_data_loader(
                opts,
            )
        else:
            self.dataloader = objects_data.imnet_quad_data_loader(opts, )

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.kp_perm = self.dataloader.dataset.kp_perm
        self.kp_perm = torch.LongTensor(self.kp_perm)
        self.preload_model_data()
        self.init_render()

    def preload_model_data(self, ):
        opts = self.opts
        model_dir, self.mean_shape, self.mean_shape_np = model_utils.load_template_shapes(
            opts
        )
        if not opts.never_deform:
            dpm, parts_data, self.kp_vertex_ids = model_utils.init_dpm(
                self.dataloader.dataset.kp_names, model_dir, self.mean_shape,
                opts.parts_file
            )
            opts.nparts = self.mean_shape['alpha'].shape[1]
            self.part_active_state, self.part_axis_init, self.part_perm = model_utils.load_active_parts(
                model_dir, self.save_dir, dpm, parts_data, suffix=''
            )

    def set_input(self, batch):
        
        input_imgs = batch['img'].type(self.Tensor)
        mask = batch['mask'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.inds = [k.item() for k in batch['inds']]
        self.imgs = input_imgs.to(self.device)
        img_size = self.imgs.shape[-1]
        mask = (mask > 0.5).float()
        self.mask = mask.to(self.device)
        codes_gt = {}
        codes_gt['img'] = self.imgs
        codes_gt['inds'] = torch.LongTensor(self.inds).to(self.device)
        codes_gt['mask'] = self.mask.unsqueeze(1)
        codes_gt['contour'] = (batch['contour']).float().to(self.device)
        codes_gt['contour'] = (codes_gt['contour'] / img_size - 0.5) * 2
        self.codes_gt = codes_gt
        

    def predict(self, ):
        opts = self.opts
        deform = not opts.never_deform
        predictions = self.model.predict(self.imgs, deform=deform)
        codes_pred = predictions
        self.codes_pred = codes_pred

        bsize = len(self.imgs)
        camera = []
        verts = []
        for b in range(bsize):
            max_ind = torch.argmax(self.codes_pred['cam_probs'][b],
                                   dim=0).item()
            camera.append(self.codes_pred['cam'][b][max_ind])
            verts.append(self.codes_pred['verts'][b][max_ind])

        camera = torch.stack(camera, )
        verts = torch.stack(verts)

        self.codes_pred['camera_selected'] = camera
        self.codes_pred['verts_selected'] = verts

    def visuals_to_save(self, count=1):
        renderer = self.renderer
        opts = self.opts
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']
        inds = self.codes_gt['inds'].data.cpu().numpy()
        uv_map = self.codes_pred['uv_map']
        camera = self.codes_pred['cam']

        if count is None:
            count = min(opts.save_visual_count, len(self.codes_gt['img']))
        visual_ids = list(range(count))
        for b in visual_ids:
            visuals = {}
            visuals['ind'] = "{:04}".format(inds[b])

            if opts.render_mask and opts.multiple_cam:
                mask_renders = self.codes_pred['mask_render'][
                    b, :, None, ...].repeat(1, 3, 1, 1).data.cpu()
                mask_renders = (mask_renders.numpy() *
                                255).astype(np.uint8).transpose(0, 2, 3, 1)
                visuals['mask_render'] = visutil.image_montage(
                    mask_renders, nrow=min(3, opts.num_hypo_cams // 3 + 1)
                )

            if opts.render_depth and opts.multiple_cam:
                all_depth_hypo = (
                    self.codes_pred['mask_render'][b] *
                    self.codes_pred['depth'][b]
                )[:, None, :, :].repeat(1, 3, 1, 1).data.cpu() / 50.0
                all_depth_hypo = (all_depth_hypo.numpy() *
                                  255).astype(np.uint8).transpose(0, 2, 3, 1)
                visuals['all_depth'] = visutil.image_montage(
                    all_depth_hypo, nrow=min(3, opts.num_hypo_cams // 3 + 1)
                )

            if opts.multiple_cam:
                vis_cam_hypotheses = renderer.render_all_hypotheses(
                    camera[b],
                    probs=self.codes_pred['cam_probs'][b],
                    verts=self.codes_pred['verts'][b]
                )
                visuals.update(vis_cam_hypotheses)

            visuals['z_img'] = visutil.tensor2im(
                visutil.undo_resnet_preprocess(img.data[b, None, :, :, :])
            )

            visuals['z_mask'] = visutil.tensor2im(
                mask.data.repeat(1, 3, 1, 1)[b, None, :, :, :]
            )
            visuals['uv_x'], visuals['uv_y'] = render_utils.render_uvmap(
                mask[b], uv_map[b].data.cpu()
            )
            # visuals['model'] = (self.render_model_using_cam(self.codes_pred['cam'][b])*255).astype(np.uint8)
            visuals['texture_copy'] = bird_vis.copy_texture_from_img(
                mask[b], img[b], self.codes_pred['project_points'][b][0]
            )

            contour_indx = (
                opts.img_size * (self.codes_gt['contour'][b] * 0.5 + 0.5)
            ).data.cpu().numpy().astype(np.int)
            visuals['contour'] = renderer.visualize_contour_img(
                contour_indx, opts.img_size
            )

            img_ix = (
                torch.FloatTensor(visuals['z_img']).permute(2, 0, 1)
            ) / 255
            visuals['a_overlay_uvmap'] = bird_vis.sample_UV_contour(
                img_ix,
                uv_map.float()[b].cpu(),
                self.sphere_uv_img,
                mask[b].float().cpu(),
                real_img=True
            )

            visuals['a_overlay_uvmap'] = visutil.tensor2im(
                [visuals['a_overlay_uvmap'].data]
            )

            self.iter_count += 1

        batch_visuals = []
        batch_visuals.append(visuals)
        return batch_visuals

    def get_current_visuals(self):
        visuals = self.visuals_to_save()[0]
        visuals.pop('ind')
        return visuals

    def test(self, ):
        opts = self.opts
        bench_stats_m1 = {
            'kps_gt': [],
            'kps_pred': [],
            'kps_err': [],
            'inds': [],
        }

        n_iter = np.min(len(self.dataloader), )
        result_path = osp.join(opts.results_dir, 'results_kp_rp.mat')
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        bench_stats = {}
        self.iter_index = None
        if not osp.exists(result_path) or opts.force_run:
            for i, batch in enumerate(tqdm(self.dataloader)):
                self.iter_index = i

                # if i % 100 == 0:
                #     print('{}/{} evaluation iterations.'.format(i, n_iter))
                self.set_input(batch)
                self.predict()
                #self.visuals_to_save(count=1)
#                visualizer.display_current_results(
#                    self.get_current_visuals(), i
#                )
                #visualizer.plot_current_points(self.get_current_points())
                visualizer.save_current_results(
                    0, self.visuals_to_save()
                )


def main(_):
    # opts.n_data_workers = 0 opts.batch_size = 1 print = pprint.pprint
    opts.batch_size = 1
    opts.results_dir = osp.join(
        opts.results_dir_base, opts.name, '%s' % (opts.split),
        'epoch_%d' % opts.num_train_epoch
    )
    opts.result_dir = opts.results_dir
    opts.dl_out_imnet = False

    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    seed = opts.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tester = UVMapTester(opts)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run(main)
