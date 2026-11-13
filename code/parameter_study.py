"""Deterministic parameter study on the same inputs; run through `make study`."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dia.common import load,gray,panels,metrics,save,segmentation,boundary,gradient,blur
from dia.annotations import wing_seeds,region_seeds,OBJECT_RECT,source_rect
from dia.scissors import marr_hildreth,link_costs,closed_path
from dia.regions import likelihood_classify,kmeans,graphcut,energy,rectangle_em
from dia.keypoints import (scale_space,extrema,localize,supported,sift_filter,
                          physical_points,affine_image,repeatability,harris,corners)
from dia.evaluation import read_seg,reference_masks,agreement,edge_agreement,expand

ROOT=Path(__file__).resolve().parent


def write_rows(out,name,rows):
    metrics(out/(name+'.json'),rows)
    # Nested values stay in JSON; CSV includes every scalar setting and measurement.
    keys=list(dict.fromkeys(k for row in rows for k,v in row.items() if not isinstance(v,(list,dict))))
    with (out/(name+'.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def heatmap(path,values,xs,ys,title,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(7,4));a=ax.imshow(values,aspect='auto',origin='lower',cmap='viridis')
    ax.set(xticks=range(len(xs)),xticklabels=xs,yticks=range(len(ys)),yticklabels=ys,
           xlabel=xlabel,ylabel=ylabel,title=title)
    for y,x in np.ndindex(values.shape):ax.text(x,y,f'{values[y,x]:.3f}',ha='center',va='center',fontsize=7,color='white' if values[y,x]<np.nanmedian(values) else 'black')
    fig.colorbar(a,ax=ax);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def q1(out):
    im=load(ROOT/'inputs/35070.jpg',320);g=gray(im);seeds=wing_seeds(im.shape)
    refs=reference_masks(ROOT/'inputs/human',im.shape)
    rb=[]
    for p in sorted((ROOT/'inputs/human').glob('*.seg')):
        labels=read_seg(p)
        labels=np.asarray(Image.fromarray(labels).resize((im.shape[1],im.shape[0]),Image.Resampling.NEAREST))
        rb.append(boundary(labels))
    sigmas=[.4,.6,.8,1.,1.4,2.,3.,4.];thresholds=[.001,.002,.005,.01,.02,.04,.08,.12,.16,.24,.32]
    rows=[];grid=np.zeros((len(sigmas),len(thresholds)))
    for i,s in enumerate(sigmas):
        for j,t in enumerate(thresholds):
            _,edges,_=marr_hildreth(g,s,t);r={'sigma':s,'threshold':t,'edge_pixels':int(edges.sum()),**edge_agreement(edges,rb)}
            grid[i,j]=r['boundary_f1'];rows.append(r)
    write_rows(out,'q1_edges',rows)
    heatmap(out/'q1_edges_grid.png',grid,thresholds,sigmas,'All-region human boundary F1 (2 px tolerance)','Slope threshold','Gaussian sigma')
    best=max(rows,key=lambda r:r['boundary_f1'])
    save(out/'q1_best_edges.png',marr_hildreth(g,best['sigma'],best['threshold'])[1])
    panels(out/'q1_edges_selected.png',[(f'Baseline: sigma=1.4, threshold=.01',marr_hildreth(g,1.4,.01)[1]),(f"Best tested: sigma={best['sigma']:g}, threshold={best['threshold']:g}",marr_hildreth(g,best['sigma'],best['threshold'])[1])],cols=2)
    panels(out/'q1_edges_examples.png',[(f"sigma={s:g}, threshold={t:g}",marr_hildreth(g,s,t)[1]) for s,t in [(1.4,.01),(best['sigma'],best['threshold']),(.4,.001),(4,.12)]],cols=2)
    configs=[{'sigma':1.4,'threshold':.01,'weights':(.43,.43,.14),'gamma':1,'seed_stride':1,'name':'Original baseline'}]
    for zi in range(5):
        for gi in range(5-zi):
            weights=(zi/4,gi/4,(4-zi-gi)/4)
            configs.append({'sigma':1.4,'threshold':.01,'weights':weights,'gamma':1,'seed_stride':1,'name':'Weight simplex'})
    for gamma in [1,2,3]:configs.append({'sigma':1.4,'threshold':.01,'weights':(.2,.7,.1),'gamma':gamma,'seed_stride':1,'name':'Gradient exponent'})
    for s,t in [(.6,.01),(1.,.01),(2.,.01),(3.,.01),(1.4,.002),(1.4,.04)]:
        configs.append({'sigma':s,'threshold':t,'weights':(.43,.43,.14),'gamma':1,'seed_stride':1,'name':'Edge parameter'})
    configs.append({'sigma':1.4,'threshold':.01,'weights':(.43,.43,.14),'gamma':1,'seed_stride':2,'name':'Sparse seeds'})
    paths=[];baseline=None;gx,gy=gradient(blur(g,1.4));m=np.hypot(gx,gy);entries=[]
    for idx,cfg in enumerate(configs):
        path,cost=closed_path(link_costs(g,cfg['weights'],cfg['sigma'],cfg['threshold'],cfg['gamma']),seeds[::cfg['seed_stride']])
        points=set(map(tuple,path))
        if baseline is None:baseline=points
        support=float(np.mean([np.mean(expand(boundary(ref))[path[:,0],path[:,1]]) for ref in refs]))
        row={**cfg,'w_zero':cfg['weights'][0],'w_gradient':cfg['weights'][1],'w_orientation':cfg['weights'][2],
             'steps':len(path)-1,'own_cost':float(cost),'baseline_jaccard':len(points&baseline)/len(points|baseline),
             'human_boundary_support':support,'mean_gradient':float(m[path[:,0],path[:,1]].mean()),'seed_count':len(seeds[::cfg['seed_stride']])}
        paths.append(row)
        np.save(out/f'q1_path_{idx:02d}.npy',path)
        if idx in [0,1,5,15,len(configs)-1]:
            def draw(ax,p=path):ax.plot(p[:,1],p[:,0],c='yellow',lw=1)
            entries.append((f"{cfg['name']}: {cfg['weights']}\nsupport={support:.3f}, seeds={row['seed_count']}",im,draw))
        print('Q1 path',idx+1,'/',len(configs),flush=True)
    write_rows(out,'q1_paths',paths);panels(out/'q1_paths_examples.png',entries,cols=2)
    return {'edge_runs':len(rows),'path_runs':len(paths),'best_edge':best,'baseline_edge':next(r for r in rows if r['sigma']==1.4 and r['threshold']==.01),'baseline_path':paths[0],
            'path_support_range':[min(r['human_boundary_support'] for r in paths),max(r['human_boundary_support'] for r in paths)]}


def q2(out):
    im=load(ROOT/'inputs/35070.jpg',320);fg,bg=region_seeds(im.shape);refs=reference_masks(ROOT/'inputs/human',im.shape)
    def score(mask):
        stats=agreement(mask,refs);free=~(fg|bg)
        stats['unmarked_iou']=float(np.mean([np.count_nonzero(mask&r&free)/max(1,np.count_nonzero((mask|r)&free)) for r in refs]))
        return stats
    ref_entries=[('Input',im)]+[(f'Human {p.stem}: insect region 1',segmentation(im,r)) for p,r in zip(sorted((ROOT/'inputs/human').glob('*.seg')),refs)]
    panels(out/'q2_references.png',ref_entries)
    regs=[.0001,.0004,.001,.002,.004,.008,.016,.032];models=[]
    for reg in regs:
        mask,costs,_=likelihood_classify(im,fg,bg,reg);models.append({'method':'Gaussian ML','regularization':reg,**score(mask)})
    for pin in [True,False]:
        hist,wcss,_=kmeans(im,fg,bg,constrain=pin)
        assert np.all(np.diff(wcss)<=1e-8)
        models.append({'method':'K-means','constrained':pin,'iterations':len(hist),'final_wcss':wcss[-1],**score(hist[-1])})
    write_rows(out,'q2_models',models)
    _,costs,_=likelihood_classify(im,fg,bg)
    lambdas=[.02,.1,.5,2.,10.,40.,160.];sigmas=[.008,.02,.04,.08,.16,.4,.8]
    cuts=[];masks=[];grid=np.zeros((len(lambdas),len(sigmas)))
    for i,lam in enumerate(lambdas):
        for j,s in enumerate(sigmas):
            mask,info=graphcut(im,costs,fg,bg,lam,s)
            assert mask[fg].all() and not mask[bg].any()
            r={'lambda':lam,'sigma':s,'energy':energy(im,costs,mask,lam,s),**info,**score(mask)}
            cuts.append(r);masks.append(mask);grid[i,j]=r['iou']
        print('Q2 lambda',lam,'done',flush=True)
    write_rows(out,'q2_graphcuts',cuts)
    heatmap(out/'q2_graphcut_grid.png',grid,sigmas,lambdas,'Mean insect IoU across five human references','Color sigma','Lambda')
    best_idx=max(range(len(cuts)),key=lambda k:cuts[k]['iou']);base_idx=next(k for k,r in enumerate(cuts) if r['lambda']==2 and r['sigma']==.08)
    indices=list(dict.fromkeys([base_idx,best_idx,0,6,42,48]))
    panels(out/'q2_graphcut_examples.png',[(f"lambda={cuts[k]['lambda']:g}, sigma={cuts[k]['sigma']:g}\nIoU={cuts[k]['iou']:.3f}, BF1={cuts[k]['boundary_f1']:.3f}",segmentation(im,masks[k])) for k in indices])
    save(out/'q2_best_graphcut.png',masks[best_idx])
    rect=source_rect(OBJECT_RECT,im.shape);ers=[];ems=[]
    for rw in [2,4,8,12]:
        for reg in regs:
            hist,stats,_,_=rectangle_em(im,rect,regularization=reg,ring_width=rw)
            ers.append({'regularization':reg,'ring_width':rw,'iterations':len(hist),'converged':stats[-1]['changed']==0,**score(hist[-1])})
            ems.append(hist[-1])
    write_rows(out,'q2_rectangles',ers)
    best_em=max(range(len(ers)),key=lambda k:ers[k]['iou']);base_em=next(k for k,r in enumerate(ers) if r['regularization']==.008 and r['ring_width']==4)
    save(out/'q2_best_rectangle.png',ems[best_em])
    panels(out/'q2_rectangle_examples.png',[(f"ridge={ers[k]['regularization']:g}, ring={ers[k]['ring_width']}\nIoU={ers[k]['iou']:.3f}",segmentation(im,ems[k])) for k in [1,base_em,best_em]],cols=3)
    return {'model_runs':len(models),'graphcut_runs':len(cuts),'rectangle_runs':len(ers),'best_graphcut':cuts[best_idx],'baseline_graphcut':cuts[base_idx],
            'best_rectangle':ers[best_em],'baseline_rectangle':ers[base_em], 'all_rectangles_converged':all(r['converged'] for r in ers)}


def detect(im,sigmas,valid=None):
    r=scale_space(gray(im),sigmas);cand=extrema(r,.005)
    d=np.column_stack([cand,abs(r[tuple(cand.T)])]);l=localize(r,cand)
    return r,supported(d,sigmas,im.shape,valid),supported(l,sigmas,im.shape,valid)


def paired_errors(d0,l0,d1,l1,sigmas,a,center,zoom,valid):
    # Same rounded scale-space cells in both methods, then same one-to-one correspondences.
    ld0={tuple(np.rint(p[:3]).astype(int)):p for p in l0};ld1={tuple(np.rint(p[:3]).astype(int)):p for p in l1}
    od=np.array([p for p in d0 if tuple(p[:3].astype(int)) in ld0]).reshape(-1,4)
    td=np.array([p for p in d1 if tuple(p[:3].astype(int)) in ld1]).reshape(-1,4)
    if not len(od) or not len(td):return {'pairs':0,'discrete_error':None,'localized_error':None}
    ol=np.array([ld0[tuple(p[:3].astype(int))] for p in od]);tl=np.array([ld1[tuple(p[:3].astype(int))] for p in td])
    pd,qd=physical_points(od,sigmas),physical_points(td,sigmas);pl,ql=physical_points(ol,sigmas),physical_points(tl,sigmas)
    for p in [pd,pl]:p[:,:2]=(p[:,:2]-center)@a.T+center;p[:,2]*=zoom
    h,w=valid.shape;candidate=[]
    for i in range(len(pd)):
        x,y,s=pl[i];m=int(np.ceil(3*s));xx,yy=int(round(x)),int(round(y))
        if xx-m<0 or xx+m>=w or yy-m<0 or yy+m>=h or not valid[yy-m:yy+m+1,xx-m:xx+m+1].all():continue
        ed=np.linalg.norm(qd[:,:2]-pd[i,:2],axis=1);el=np.linalg.norm(ql[:,:2]-pl[i,:2],axis=1)
        rd=np.maximum(qd[:,2]/pd[i,2],pd[i,2]/qd[:,2]);rl=np.maximum(ql[:,2]/pl[i,2],pl[i,2]/ql[:,2])
        for j in np.flatnonzero((ed<=3)&(el<=3)&(rd<=np.sqrt(2))&(rl<=np.sqrt(2))):candidate.append(((ed[j]+el[j])/2,i,int(j),float(ed[j]),float(el[j])))
    ui=set();uj=set();errors=[]
    for _,i,j,ed,el in sorted(candidate):
        if i not in ui and j not in uj:ui.add(i);uj.add(j);errors.append((ed,el))
    return {'pairs':len(errors),'discrete_error':float(np.mean(errors,axis=0)[0]) if errors else None,
            'localized_error':float(np.mean(errors,axis=0)[1]) if errors else None}


def q3(out):
    im=load(ROOT/'inputs/118035.jpg',400);sigmas=1.1*2**(np.arange(11)/2)
    original,disc,loc=detect(im,sigmas)
    transforms=[('Identity',0,1,1,0,0,4),('Rotation only',22,1,1,0,0,4),('Scale only',0,.82,1,0,0,4),
                ('Intensity only',0,1,.85,.07,0,4),('Noise only',0,1,1,0,.008,4),
                ('Combined seed 4',22,.82,.85,.07,.008,4),('Combined seed 17',22,.82,.85,.07,.008,17),
                ('Combined seed 29',22,.82,.85,.07,.008,29),('Harder combined',37,.65,.7,.1,.016,4)]
    contrasts=[.008,.015,.025,.04,.06,.08,.12,.18];ratios=[2,3,5,8,10,15,20,30]
    data=[];count_grid=np.zeros((len(contrasts),len(ratios)))
    selections={}
    for i,t in enumerate(contrasts):
        for j,r in enumerate(ratios):
            p=sift_filter(original,loc,t,r);selections[t,r]=p;count_grid[i,j]=len(p)
    heatmap(out/'q3_counts_grid.png',count_grid,ratios,contrasts,'Original supported keypoint count','Edge ratio r','Contrast T')
    paired=[];by_transform=[];examples=[]
    for name,angle,zoom,gain,offset,noise,seed in transforms:
        modified,a,center,valid=affine_image(im,angle,zoom,gain,offset,noise,seed)
        response,md,ml=detect(modified,sigmas,valid)
        for t in contrasts:
            for ratio in ratios:
                p=selections[t,ratio];q=sift_filter(response,ml,t,ratio)
                _,stats=repeatability(physical_points(p,sigmas),physical_points(q,sigmas),a,center,zoom,valid)
                data.append({'transform':name,'contrast':t,'edge_ratio':ratio,'original_count':len(p),'modified_count':len(q),**stats})
        p=sift_filter(original,disc,.025,10);q=sift_filter(response,md,.025,10)
        _,stats=repeatability(physical_points(p,sigmas),physical_points(q,sigmas),a,center,zoom,valid)
        by_transform.append({'transform':name,'method':'Discrete',**stats})
        lp=selections[.025,10];lq=sift_filter(response,ml,.025,10)
        _,stats=repeatability(physical_points(lp,sigmas),physical_points(lq,sigmas),a,center,zoom,valid)
        by_transform.append({'transform':name,'method':'Localized',**stats})
        paired.append({'transform':name,**paired_errors(p,lp,q,lq,sigmas,a,center,zoom,valid)})
        if name in ['Rotation only','Scale only','Noise only','Harder combined']:
            pp=physical_points(lq,sigmas)
            def draw(ax,points=pp):
                from matplotlib.patches import Circle
                for x,y,s in points:ax.add_patch(Circle((x,y),np.sqrt(2)*s,fill=False,edgecolor='cyan',lw=.6))
            examples.append((f"{name}\nrepeatability={stats['repeatability']:.3f}",modified,draw))
        print('Q3',name,'done',flush=True)
    write_rows(out,'q3_thresholds',data);write_rows(out,'q3_transforms',by_transform);write_rows(out,'q3_paired',paired)
    panels(out/'q3_transform_examples.png',examples,cols=2)
    tuning=[]
    for t in contrasts:
        for ratio in ratios:
            rows=[r for r in data if r['contrast']==t and r['edge_ratio']==ratio and r['transform'].startswith('Combined')]
            tuning.append({'contrast':t,'edge_ratio':ratio,'original_count':len(selections[t,ratio]),
                           'mean_repeatability':float(np.mean([r['repeatability'] for r in rows])),
                           'min_repeatability':min(r['repeatability'] for r in rows),
                           'mean_matches':float(np.mean([r['matches'] for r in rows])),
                           'mean_error_px':float(np.mean([r['mean_error_px'] for r in rows if r['mean_error_px'] is not None]))})
    write_rows(out,'q3_threshold_summary',tuning)
    heatmap(out/'q3_repeatability_grid.png',np.array([r['mean_repeatability'] for r in tuning]).reshape(len(contrasts),len(ratios)),ratios,contrasts,'Mean repeatability: combined transform, three noise seeds','Edge ratio r','Contrast T')
    hrows=[]
    for ds in [.6,1.,1.6]:
        for ws in [1.,1.8,3.]:
            for k in [.03,.04,.06]:
                _,_,r=harris(gray(im),ds,ws,k)
                for fraction in [.005,.015,.04]:
                    pts=corners(r,fraction)
                    hrows.append({'derivative_sigma':ds,'window_sigma':ws,'harris_k':k,'threshold_fraction':fraction,'corners':len(pts)})
    write_rows(out,'q3_harris',hrows)
    ladders=[]
    for per_octave in [2,3,4]:
        ss=1.1*2**(np.arange(5*per_octave+1)/per_octave)
        r,d,l=detect(im,ss);mod,a,center,valid=affine_image(im);rr,dd,ll=detect(mod,ss,valid)
        p=sift_filter(r,l,.025,10);q=sift_filter(rr,ll,.025,10)
        _,stats=repeatability(physical_points(p,ss),physical_points(q,ss),a,center,.82,valid)
        ladders.append({'scales_per_octave':per_octave,'scale_count':len(ss),'original_count':len(p),'modified_count':len(q),**stats})
    write_rows(out,'q3_scale_ladders',ladders)
    return {'threshold_pairs':len(tuning),'threshold_transform_runs':len(data),'transform_cases':len(transforms),'harris_runs':len(hrows),'scale_ladders':ladders,
            'baseline_threshold':next(r for r in tuning if r['contrast']==.025 and r['edge_ratio']==10),
            'best_repeatability_with_100_points':max((r for r in tuning if r['original_count']>=100),key=lambda r:r['mean_repeatability']),
            'combined_paired':next(r for r in paired if r['transform']=='Combined seed 4')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['q1','q2','q3','all'],default='all');ap.add_argument('--outdir',type=Path,default=ROOT/'outputs/study');args=ap.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    summary={}
    for name,fn in [('q1',q1),('q2',q2),('q3',q3)]:
        if args.part in [name,'all']:
            summary[name]=fn(args.outdir);metrics(args.outdir/(name+'_summary.json'),summary[name])
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
