"""CPU-only front attachment tests; no fitter or CUDA initialization."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from fit_session import fit_input, conventional_input_paths, ScrollSpec
from front_points import get_front_attachment_loss, load_front_points


class Scaling(torch.nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, p):
        return p * self.scale

    def inv(self, p):
        return p / self.scale


class FrontPointsTests(unittest.TestCase):
    def loss(self, points, scale=1., domain=(1, 5)):
        transform = Scaling(scale)
        value = get_front_attachment_loss(transform, torch.tensor(10., requires_grad=True),
                                         torch.tensor(points, dtype=torch.float32).reshape(-1,3),
                                         64, domain)
        return value, transform

    def test_zero_on_unlabeled_windings(self):
        value, transform = self.loss([[0, 0, 10], [0, 0, 20], [0, 0, 40]])
        self.assertEqual(value.item(), 0.)
        value.backward()
        self.assertTrue(torch.isfinite(transform.scale.grad))

    def test_distance_in_scan_space_and_gradient(self):
        value, transform = self.loss([[0, 0, 10]], scale=1.2)
        self.assertAlmostEqual(value.item(), 10-10/1.2-0.5, places=5)
        value.backward()
        self.assertGreater(transform.scale.grad.item(), 0)
        before = value.item()
        with torch.no_grad():
            transform.scale -= 0.001*transform.scale.grad
        after = get_front_attachment_loss(transform,torch.tensor(10.),torch.tensor([[0.,0.,10.]]),1,(1,5))
        self.assertLess(after.item(), before)

    def test_between_windings_and_domain_edges(self):
        self.assertAlmostEqual(self.loss([[0,0,15]])[0].item(),4.5)
        self.assertAlmostEqual(self.loss([[0,0,0]])[0].item(),9.5)
        self.assertAlmostEqual(self.loss([[0,0,60]])[0].item(),9.5)

    def test_branch_cut_and_missing_observations(self):
        points = []
        for theta in (-1e-4,1e-4):
            r = 30 + 10*theta/(2*np.pi)
            points.append([0,r*np.sin(theta),r*np.cos(theta)])
        self.assertLess(self.loss(points)[0].item(),1e-9)
        self.assertEqual(self.loss([[0,0,40]])[0].item(),0.)
        empty, _ = self.loss([])
        empty.backward()
        self.assertEqual(empty.item(),0.)
        self.assertEqual(self.loss([[0,0,10]],domain=(1,0))[0].item(),0.)

    def test_optional_catalog(self):
        spec=fit_input('front_points')
        self.assertFalse(spec.enabled({}))
        self.assertFalse(spec.required(Config().as_dict()))
        fields=Config.catalog()['schema']['fields']
        self.assertEqual(fields['input_use_front_points']['runtime_impact'],'new_fit')
        self.assertTrue(fields['sample_count_front_points']['scale_with_z'])
        cfg={'input_use_front_points':True,'loss_weight_front_attachment':1.}
        self.assertTrue(spec.enabled(cfg)); self.assertTrue(spec.required(cfg))
        self.assertFalse(spec.enabled({**cfg,'input_use_front_points':False}))
        self.assertIn('front_points',conventional_input_paths('/tmp/data',ScrollSpec(name='test',voxel_size_um=1.,spiral_outward_sense='CW')).manifest())
        with self.assertRaises(ValueError): Config({'front_attachment_huber_delta':0.})

    def test_loader_filter_validation_and_identity(self):
        with tempfile.TemporaryDirectory() as root:
            group = zarr.open_group(root,mode='w',zarr_format=2)
            metadata={'artifact_type':'spiral_front_points','format_version':1,
                      'coordinate_order':'zyx','coordinate_units':'working_voxels',
                      'working_voxel_size_um':9.362,'origin_mm_xyz':[0,0,0],
                      'shape_zyx':[100,100,100]}
            group.attrs.update(metadata)
            a=group.create_array('position_zyx',data=np.array([[5,2,3],[15,2,3]],dtype=np.float32))
            points, fingerprint=load_front_points(root,0,10)
            np.testing.assert_array_equal(points,[[5,2,3]])
            self.assertEqual(load_front_points(root,0,20)[1],fingerprint)
            self.assertEqual(len(load_front_points(root,30,40)[0]),0)
            group.attrs['field_experiment']='moved'
            self.assertEqual(load_front_points(root,0,10)[1],fingerprint)
            a[0,1]=4
            self.assertNotEqual(load_front_points(root,0,10)[1],fingerprint)
            a[0,1]=float('nan')
            with self.assertRaises(ValueError): load_front_points(root,0,10)

    def test_existing_normal_decoder(self):
        # Load the actual two decoder definitions without importing the fitter's
        # unrelated native/ODE dependencies into this CPU interchange test.
        tree=ast.parse((Path(__file__).resolve().parents[1]/'losses.py').read_text())
        definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                     and node.name in ('_decode_uint8_normal_component','_decode_uint8_normal')]
        namespace={'torch':torch,'F':F}
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'losses.py','exec'),namespace)
        normal,valid=namespace['_decode_uint8_normal'](torch.tensor([255,128,0]),torch.tensor([128,128,0]))
        torch.testing.assert_close(normal[:2],torch.tensor([[0.,0.,1.],[1.,0.,0.]]))
        torch.testing.assert_close(valid,torch.tensor([1.,1.,0.]))


    @unittest.skipUnless(os.environ.get('EISODOS_EXPORT_TEST_DIR'),
                         'set EISODOS_EXPORT_TEST_DIR to a retained Julia test export')
    def test_julia_interchange(self):
        root=Path(os.environ['EISODOS_EXPORT_TEST_DIR'])/'export'
        points,_=load_front_points(root/'front_points.zarr',0,100)
        self.assertEqual(points.shape,(2000,3))
        self.assertTrue(np.all((points[:,0]>=8-1e-4) & (points[:,0]<=27+1e-4)))
        nx=zarr.open_group(str(root/'normal_x.zarr'),mode='r')['0']
        ny=zarr.open_group(str(root/'normal_y.zarr'),mode='r')['0']
        self.assertEqual(nx.shape,(20,18,16))
        self.assertEqual(nx[0,0,0],0)
        self.assertTrue(np.any(nx[:]!=0))
        self.assertEqual(nx.shape,ny.shape)
        # Exercise the same CPU packer the existing normal loader uses,
        # including Julia's compressor and full-sized boundary chunks.
        from pack_resident_pools import pack_arrays, verify_pool
        with tempfile.TemporaryDirectory() as packed:
            pack_arrays([str(root/'normal_x.zarr'/'0'),str(root/'normal_y.zarr'/'0')],
                        packed,label='CPU normal interchange',io_threads=1)
            verify_pool(packed,2000)
        tree=ast.parse((Path(__file__).resolve().parents[1]/'losses.py').read_text())
        definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                     and n.name in ('_decode_uint8_normal_component','_decode_uint8_normal')]
        ns={'torch':torch,'F':F}
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'losses.py','exec'),ns)
        fixture=json.loads((root/'normal-roundtrip.json').read_text())
        normals=np.asarray(fixture['normals']); encoded=torch.tensor(fixture['encoded'])
        decoded,valid=ns['_decode_uint8_normal'](encoded[:,0],encoded[:,1])
        decoded=decoded.numpy()[:,::-1]
        dots=np.abs((decoded*normals).sum(axis=1)/(np.linalg.norm(decoded,axis=1)*np.linalg.norm(normals,axis=1)))
        error=np.rad2deg(np.arccos(np.clip(dots,-1,1)))
        self.assertTrue(valid.bool().all())
        self.assertLess(error.max(),7.)
        self.assertLess(np.median(error[:10000]),0.4)


if __name__ == '__main__':
    unittest.main()
