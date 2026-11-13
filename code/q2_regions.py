"""Q2: Gaussian ML, constrained k-means, binary graph cuts and rectangle EM."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dia.common import load,panels,save,metrics,segmentation
from dia.regions import likelihood_classify,kmeans,graphcut,energy,rectangle_em


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',default='inputs/35070.jpg');ap.add_argument('--size',type=int,default=320);ap.add_argument('--outdir',default='outputs/q2');args=ap.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True);im=load(args.image,args.size);h,w=im.shape[:2]
    # Rectangles (x0,y0,x1,y1), manually selected in original source coordinates.
    fg_rects=[(95,132,225,158),(150,160,183,171)]
    bg_rects=[(15,15,450,70),(15,275,450,315),(380,225,470,265),(445,110,475,215)]
    def rects_mask(rects):
        mask=np.zeros((h,w),bool)
        for r in rects:
            x0,y0,x1,y1=np.rint(np.array(r)*[w/481,h/321,w/481,h/321]).astype(int);mask[y0:y1,x0:x1]=True
        return mask
    fg,bg=rects_mask(fg_rects),rects_mask(bg_rects)
    training=im.copy();training[fg]=[1,.1,.1];training[bg]=[.1,.4,1]
    labels,costs,models=likelihood_classify(im,fg,bg)
    likes=np.exp(-(costs-costs.min((0,1),keepdims=True)))
    panels(out/'likelihood.png',[('Marked FG (red), BG (blue)',training),('Foreground likelihood (unit peak)',likes[:,:,1]),('Background likelihood (unit peak)',likes[:,:,0]),('Gaussian maximum likelihood',segmentation(im,labels))],cols=2)
    save(out/'ml_mask.png',labels);history,wcss,means=kmeans(im,fg,bg)
    indices=list(dict.fromkeys([0,min(1,len(history)-1),min(2,len(history)-1),len(history)-1]))
    panels(out/'kmeans.png',[(f'K-means iteration {i+1}',segmentation(im,history[i])) for i in indices],cols=2)
    fig,ax=plt.subplots(figsize=(6,3));ax.plot(np.arange(1,len(wcss)+1),wcss,'o-');ax.set(xlabel='Iteration (after mean update)',ylabel='WCSS',title='Constrained two-class k-means');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'wcss.png',dpi=150);plt.close(fig)
    entries=[];cuts=[]
    for lam,sigma in [(2,.08),(.02,.08),(40,.08),(2,.008),(2,.8)]:
        mask,info=graphcut(im,costs,fg,bg,lam,sigma)
        assert mask[fg].all() and not mask[bg].any()
        e=energy(im,costs,mask,lam,sigma)
        cuts.append({'lambda':lam,'sigma':sigma,'foreground_pixels':int(mask.sum()),'energy':e,**info})
        entries.append((f'lambda={lam:g}, sigma={sigma:g}',segmentation(im,mask)))
        if lam==2 and sigma==.08:save(out/'graphcut_mask.png',mask)
    panels(out/'graphcuts.png',entries,cols=3)
    rect=tuple(np.rint(np.array([35,80,450,270])*[w/481,h/321,w/481,h/321]).astype(int))
    hist,stats,inner,ring=rectangle_em(im,rect)
    def draw(ax):
        x0,y0,x1,y1=rect;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='yellow',lw=1.5))
    entries=[('Only rectangle supplied',im,draw)]
    init=im.copy();init[ring]=[0,.3,1];entries.append(('Initial BG boundary ring (4 px)',init))
    for i in list(dict.fromkeys([0,1,2,len(hist)-1])):
        if i<len(hist):entries.append((f'Gaussian/ML iteration {i+1}',segmentation(im,hist[i])))
    panels(out/'rectangle.png',entries,cols=3)
    save(out/'rectangle_mask.png',hist[-1])
    metrics(out/'metrics.json',{'image':args.image,'shape':list(im.shape),'fg_rects_source_xyxy':fg_rects,'bg_rects_source_xyxy':bg_rects,'foreground_training_pixels':int(fg.sum()),'background_training_pixels':int(bg.sum()),'models':[{'mean':m.tolist(),'covariance':c.tolist()} for m,c in models],'ml_foreground_pixels':int(labels.sum()),'kmeans_wcss':wcss,'kmeans_means':means.tolist(),'kmeans_converged':len(history)<100,'graphcuts':cuts,'rectangle_xyxy':[int(x) for x in rect],'rectangle_regularization':.008,'rectangle_iterations':stats,'rectangle_converged':stats[-1]['changed']==0})
    print('Q2 complete; k-means iterations',len(history),'rectangle iterations',len(hist),flush=True)

if __name__=='__main__': main()
