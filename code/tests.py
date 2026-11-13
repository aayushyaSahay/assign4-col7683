"""Independent references and analytic invariants for the numerical algorithms."""
import itertools
import unittest
import numpy as np
from dia.common import blur,laplace,gradient
from dia.scissors import marr_hildreth,link_costs,dijkstra,trace
from dia.regions import fit_gaussian,nll,kmeans,graphcut,energy,rectangle_em
from dia.keypoints import derivatives3,localize,extrema,harris,corners,affine_image,repeatability


class Algorithms(unittest.TestCase):
    def test_smoothing_constant_and_impulse(self):
        a=np.ones((31,31));np.testing.assert_allclose(blur(a,2),1,atol=1e-14)
        a[:]=0;a[15,15]=1;b=blur(a,1.5)
        self.assertAlmostEqual(b.sum(),1);np.testing.assert_allclose(b,b.T,atol=1e-15)
        np.testing.assert_allclose(laplace(np.ones((5,7))),0)

    def test_derivatives_polynomial(self):
        y,x=np.mgrid[:9,:11];gx,gy=gradient(x+3*y)
        np.testing.assert_allclose(gx[1:-1,1:-1],1);np.testing.assert_allclose(gy[1:-1,1:-1],3)
        np.testing.assert_allclose(laplace(x*x+y*y)[1:-1,1:-1],4)

    def test_marr_step(self):
        a=np.zeros((31,31));a[:,16:]=1
        r,e,s=marr_hildreth(a,1.4,.01)
        self.assertTrue(e[:,14:18].any());self.assertFalse(e[:,:10].any())
        self.assertFalse(marr_hildreth(np.ones((20,20)))[1].any())
        self.assertTrue(np.all(marr_hildreth(a,1.4,.02)[1]<=e))

    def test_dijkstra_against_bellman_ford(self):
        im=np.random.default_rng(2).random((4,5));c=link_costs(im)
        dist,parent=dijkstra(c,(1,1));ref=np.full(im.shape,np.inf);ref[1,1]=0
        from dia.scissors import NEIGHBORS
        for _ in range(im.size-1):
            for y,x in np.ndindex(im.shape):
                for k,(dy,dx) in enumerate(NEIGHBORS):
                    yy,xx=y+dy,x+dx
                    if 0<=yy<4 and 0<=xx<5:ref[yy,xx]=min(ref[yy,xx],ref[y,x]+c[k,y,x])
        np.testing.assert_allclose(dist,ref)
        path=trace(parent,(1,1),(3,4));self.assertEqual(tuple(path[0]),(1,1));self.assertEqual(tuple(path[-1]),(3,4))
        self.assertTrue((abs(np.diff(path,axis=0)).max(axis=1)==1).all())

    def test_dijkstra_grid_distance(self):
        c=np.ones((8,5,5));dist,_=dijkstra(c,(0,0))
        y,x=np.mgrid[:5,:5];np.testing.assert_allclose(dist,np.maximum(y,x))

    def test_gaussian_likelihood_analytic(self):
        model=(np.zeros(3),np.eye(3));x=np.array([[0,0,0],[1,2,3.]])
        np.testing.assert_allclose(nll(x,model),1.5*np.log(2*np.pi)+np.array([0,7]))
        m,c=fit_gaussian(np.ones((4,3)));self.assertTrue((np.linalg.eigvalsh(c)>0).all())

    def test_kmeans_monotonic(self):
        rng=np.random.default_rng(1);im=rng.random((12,13,3));fg=np.zeros((12,13),bool);bg=fg.copy();fg[1,1]=True;bg[-2,-2]=True
        hist,wcss,_=kmeans(im,fg,bg)
        self.assertTrue((np.diff(wcss)<=1e-10).all());self.assertTrue(all(m[1,1] and not m[-2,-2] for m in hist))
        self.assertTrue(np.array_equal(hist[-1],hist[-2]))

    def test_graphcut_exhaustive(self):
        rng=np.random.default_rng(7)
        for trial in range(5):
            im=rng.random((2,3,3));cost=rng.random((2,3,2))*2
            fg=np.zeros((2,3),bool);bg=fg.copy();fg[0,0]=True;bg[1,2]=True
            labels,info=graphcut(im,cost,fg,bg,1,.7,scale=10000)
            all_e=[]
            for bits in itertools.product([False,True],repeat=6):
                lab=np.array(bits).reshape(2,3)
                if lab[0,0] and not lab[1,2]:all_e.append(energy(im,cost,lab,1,.7))
            self.assertLessEqual(abs(energy(im,cost,labels,1,.7)-min(all_e)),.002)
            self.assertTrue(labels[0,0] and not labels[1,2])

    def test_rectangle_only_separation(self):
        im=np.empty((32,32,3));im[:]=[.4,.7,.2];im[10:22,10:22]=[.05,.08,.04]
        history,stats,inner,ring=rectangle_em(im,(5,5,27,27))
        expected=np.zeros((32,32),bool);expected[10:22,10:22]=True
        np.testing.assert_array_equal(history[-1],expected)
        self.assertEqual(stats[-1]['changed'],0)

    def test_taylor_localization_known_peak(self):
        s,y,x=np.mgrid[:7,:9,:11];a=10-(s-3.2)**2-.5*(y-4.3)**2-.8*(x-5.15)**2
        # Localizer accepts a positive LoG maximum with a known fractional offset.
        out=localize(a,np.array([[3,4,5]]));np.testing.assert_allclose(out[0,:3],[3.2,4.3,5.15],atol=1e-12)
        self.assertAlmostEqual(out[0,3],10)
        self.assertEqual(len(extrema(a,.1)),1)

    def test_scale_extrema_strict(self):
        a=np.zeros((5,9,9));a[2,4,4]=-2
        np.testing.assert_array_equal(extrema(a,1),[[2,4,4]])
        a[2,4,5]=-2;self.assertEqual(len(extrema(a,1)),0)

    def test_harris_flat_and_corner(self):
        _,_,r=harris(np.ones((25,25)));np.testing.assert_allclose(r,0,atol=1e-20)
        a=np.zeros((41,41));a[20:,20:]=1;det,tr,r=harris(a)
        pts=corners(r);self.assertLess(np.linalg.norm(pts-[20,20],axis=1).min(),4)
        self.assertTrue((det>=-1e-15).all());self.assertTrue((tr>=0).all())

    def test_affine_identity_and_geometry(self):
        im=np.random.default_rng(1).random((20,30,3));out,a,c,v=affine_image(im,0,1,1,0,0)
        np.testing.assert_allclose(out[v],im[v],atol=1e-14)
        p=np.array([[15.,10.,1.],[13.,9.,1.]])
        expected,stats=repeatability(p,p,a,c,1,np.ones((20,30),bool))
        self.assertEqual(stats['matches'],2);self.assertEqual(stats['repeatability'],1)

if __name__=='__main__': unittest.main(verbosity=2)
