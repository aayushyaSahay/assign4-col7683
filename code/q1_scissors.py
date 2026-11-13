"""Q1: Marr-Hildreth, four Dijkstra metrics, closed intelligent-scissors paths."""
from pathlib import Path
import argparse
import numpy as np
from dia.common import load,gray,panels,save,metrics
from dia.annotations import wing_seeds
from dia.scissors import marr_hildreth,link_costs,dijkstra,closed_path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',default='inputs/35070.jpg');ap.add_argument('--size',type=int,default=320);ap.add_argument('--outdir',default='outputs/q1');args=ap.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    im=load(args.image,args.size);g=gray(im);save(out/'original.png',im)
    seeds=wing_seeds(im.shape)
    def marks(ax):
        a=np.array(seeds);ax.scatter(a[:,1],a[:,0],s=14,c='red')
        for i,(y,x) in enumerate(a):ax.text(x+2,y-2,str(i+1),fontsize=7,color='white')
    entries=[('Input and manually chosen wing seeds',im,marks)];sweeps=[]
    for sigma,t in [(.6,.01),(1.4,.01),(3.,.01),(1.4,.002),(1.4,.04)]:
        response,edges,strength=marr_hildreth(g,sigma,t);entries.append((f'sigma={sigma:g}, slope threshold={t:g}',edges))
        sweeps.append({'sigma':sigma,'threshold':t,'edge_pixels':int(edges.sum())})
    panels(out/'edges.png',entries)
    maps=[];ds={};seed=tuple(seeds[0])
    for name,weights in [('Zero crossings',(1,0,0)),('Gradient magnitude',(0,1,0)),('Gradient orientation',(0,0,1)),('Combined (.43,.43,.14)',(.43,.43,.14))]:
        costs=link_costs(g,weights);dist,_=dijkstra(costs,seed)
        maps.append((name+f'\nmax distance={dist.max():.2f}',dist/dist.max()))
        ds[name]={'max':float(dist.max()),'mean':float(dist.mean())}
        np.save(out/(name.split()[0].lower()+'_distance.npy'),dist)
    panels(out/'distances.png',maps,cols=2)
    entries=[];paths={};base=None
    for name,w,gamma in [('Baseline',(.43,.43,.14),1),('Orientation emphasized',(.2,.2,.6),1),('Stronger gradient selectivity',(.2,.7,.1),2)]:
        path,cost=closed_path(link_costs(g,w,gamma=gamma),seeds)
        def draw(ax,p=path):
            ax.plot(p[:,1],p[:,0],color='yellow',lw=1.2);marks(ax)
        entries.append((name,im,draw))
        if base is None:base=set(map(tuple,path))
        points=set(map(tuple,path));paths[name]={'weights':w,'gradient_exponent':gamma,'path_pixels':len(path)-1,'cost':cost,'overlap_with_baseline':len(points&base)/len(points|base)}
        np.save(out/(name.split()[0].lower()+'_path.npy'),path)
    panels(out/'paths.png',entries)
    metrics(out/'metrics.json',{'image':args.image,'shape':list(im.shape),'seeds_yx':seeds,'edge_sweeps':sweeps,'distance_maps':ds,'paths':paths})
    print('Q1 complete',paths,flush=True)

if __name__=='__main__': main()
