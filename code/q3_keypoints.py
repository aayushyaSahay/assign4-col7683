"""Q3: normalized LoG blobs, Taylor localization, Harris and SIFT criteria."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dia.common import load,gray,panels,save,metrics
from dia.keypoints import scale_space,extrema,localize,harris,corners,sift_filter,affine_image,physical_points,repeatability,supported


def overlay(points,sigmas,color='yellow'):
    physical=physical_points(points,sigmas)
    def draw(ax):
        for x,y,s in physical:ax.add_patch(Circle((x,y),np.sqrt(2)*s,fill=False,edgecolor=color,lw=.65))
    return draw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',default='inputs/118035.jpg');ap.add_argument('--size',type=int,default=400);ap.add_argument('--outdir',default='outputs/q3');args=ap.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True);im=load(args.image,args.size);g=gray(im);save(out/'original.png',im)
    sigmas=1.1*2**(np.arange(11)/2);responses=scale_space(g,sigmas)
    candidates=extrema(responses,.005);localized=localize(responses,candidates)
    discrete=np.column_stack([candidates,abs(responses[tuple(candidates.T)])])
    localized=supported(localized,sigmas,im.shape)
    discrete=supported(discrete,sigmas,im.shape)
    panels(out/'blobs.png',[(f'Discrete scale-space maxima: {len(discrete)}',im,overlay(discrete,sigmas)),(f'Taylor localized maxima: {len(localized)}',im,overlay(localized,sigmas))],cols=2)
    entries=[(f'sigma={sigmas[i]:.2f}',abs(responses[i])) for i in [0,2,4,6,8,10]]
    panels(out/'scale_space.png',entries)
    det,tr,r=harris(g);cs=corners(r)
    def draw_cs(ax):ax.scatter(cs[:,1],cs[:,0],s=8,facecolors='none',edgecolors='yellow',linewidths=.7)
    # Symmetric signed-log display retains sign and faint responses.
    signed=lambda a:np.sign(a)*np.log1p(abs(a)/(max(abs(a).max(),1e-12)*1e-4))
    panels(out/'harris.png',[('Structure tensor determinant (log)',signed(det)),('Structure tensor trace (log)',signed(tr)),('Harris response (signed log)',signed(r)),(f'Harris corners: {len(cs)}',im,draw_cs)],cols=2)
    entries=[];counts={}
    for c in [.008,.08,.18]:
        pts=sift_filter(responses,localized,c,1e10);counts[f'contrast_{c}']=len(pts)
        entries.append((f'Contrast >= {c:g}: {len(pts)}',im,overlay(pts,sigmas)))
    for ratio in [3,10,30]:
        pts=sift_filter(responses,localized,0,ratio);counts[f'edge_ratio_{ratio}']=len(pts)
        entries.append((f'Edge ratio r={ratio}: {len(pts)}',im,overlay(pts,sigmas)))
    panels(out/'criteria.png',entries)
    final=sift_filter(responses,localized,.025,10)
    panels(out/'selected.png',[(f'Both criteria: contrast=.025, r=10 ({len(final)})',im,overlay(final,sigmas))],cols=1,figsize=(8,5))
    modified,a,center,valid=affine_image(im)
    mr=scale_space(gray(modified),sigmas);mp=localize(mr,extrema(mr,.005));mf=supported(sift_filter(mr,mp,.025,10),sigmas,im.shape,valid)
    physical=physical_points(final,sigmas);detected=physical_points(mf,sigmas)
    expected,stats=repeatability(physical,detected,a,center,.82,valid)
    md=extrema(mr,.005)
    md=np.column_stack([md,abs(mr[tuple(md.T)])])
    fd=sift_filter(responses,discrete,.025,10);mfd=supported(sift_filter(mr,md,.025,10),sigmas,im.shape,valid)
    _,discrete_stats=repeatability(physical_points(fd,sigmas),physical_points(mfd,sigmas),a,center,.82,valid)
    def compare(ax):
        overlay(mf,sigmas,'cyan')(ax)
        for x,y,s in expected:ax.add_patch(Circle((x,y),np.sqrt(2)*s,fill=False,edgecolor='magenta',lw=.8,linestyle='--'))
    panels(out/'transformed.png',[('Original selected keypoints',im,overlay(final,sigmas)),('Detected: cyan; transformed original: magenta',modified,compare)],cols=2)
    save(out/'modified.png',modified);np.save(out/'keypoints.npy',physical);np.save(out/'modified_keypoints.npy',detected)
    # Localization shift is descriptive, not a claim of error improvement without ground truth.
    shifts=[]
    for p in localized:
        distances=np.linalg.norm(candidates-p[:3],axis=1);shifts.append(float(distances.min()))
    metrics(out/'metrics.json',{'image':args.image,'shape':list(im.shape),'sigmas':sigmas.tolist(),'discrete_maxima':len(discrete),'localized_maxima':len(localized),'mean_localization_shift_index_y_x':float(np.mean(shifts)),'harris_corners':len(cs),'criteria_counts':counts,'selected_keypoints':len(final),'modified_keypoints':len(mf),'transform':{'angle_degrees':22,'scale':.82,'gain':.85,'offset':.07,'noise_std':.008,'noise_seed':4,'matrix':a.tolist(),'center':center.tolist()},'repeatability':stats,'discrete_repeatability':discrete_stats})
    print('Q3 complete',stats,flush=True)

if __name__=='__main__': main()
