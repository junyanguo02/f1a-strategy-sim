#!/usr/bin/env python3
"""T051：用 Pillow 给 T048 图加标题/坐标轴标题/色标（matplotlib 字体栈在本机不可用）。
只写 analysis/dirty_air/figures/；不改 T048 原图。"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import findfont
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/track_geometry/outputs'
DST=ROOT/'analysis/dirty_air/figures'
FONT=findfont('DejaVu Sans')

def make(rid):
    d=pd.read_parquet(OUT/f'{rid}.parquet')
    x=d.x_m.to_numpy(); y=d.y_m.to_numpy(); s=d.s_m.to_numpy(); k=d.curvature_1pm.to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    ax=axes[0]; norm=plt.Normalize(vmin=float(np.percentile(k,1)),vmax=float(np.percentile(k,99)))
    for i in range(len(x)):
        j=(i+1)%len(x)
        ax.plot([x[i],x[j]],[y[i],y[j]],color=plt.cm.coolwarm(norm(k[i])),lw=2.0)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax2=axes[1]; ax2.plot(s,k,lw=1.2); ax2.axhline(0,color='gray',lw=.6); ax2.set_xticks([]); ax2.set_yticks([])
    fig.subplots_adjust(left=.07,right=.88,bottom=.13,top=.88,wspace=.10)
    tmp=Path('/tmp')/f't048_{rid}_notext.png'; fig.savefig(tmp,dpi=150); plt.close(fig)
    im=Image.open(tmp).convert('RGB'); W,H=im.size; draw=ImageDraw.Draw(im)
    def font(sz): return ImageFont.truetype(FONT,sz)
    f_title=font(26); f_label=font(20); f_tick=font(16); f_panel=font(20)
    title=f'{rid}  {d.shape[0]} points, 5 m grid'
    w=draw.textlength(title,font=f_title); draw.text(((W-w)/2,8),title,fill=(20,20,20),font=f_title)
    draw.text((W*0.26, H-38),'x (m)',fill=(30,30,30),font=f_label,anchor='mm')

    draw.text((W*0.68, H-38),'s (m)',fill=(30,30,30),font=f_label,anchor='mm')

    # 面板小标题
    draw.text((W*0.25, H*0.90),'track, colour = signed curvature',fill=(30,30,30),font=f_panel,anchor='mm')
    draw.text((W*0.68, H*0.90),'curvature vs distance',fill=(30,30,30),font=f_panel,anchor='mm')
    # 旋转的 y / curvature 标题
    def paste_rot(text,xc,yc,fp):
        tmp=Image.new('RGBA',(260,34),(255,255,255,0)); d3=ImageDraw.Draw(tmp)
        d3.text((0,0),text,fill=(30,30,30),font=fp)
        rot=tmp.rotate(90,expand=True)
        im.paste(rot,(int(xc),int(yc)-rot.size[1]//2),rot)
    paste_rot('y (m)',24,int(H*0.50),f_label)
    paste_rot('curvature (1/m)',int(W*0.445),int(H*0.50),f_label)
    # 色标
    x0=int(W*0.905); x1=int(W*0.935); y0=int(H*0.16); y1=int(H*0.84)
    vals=np.linspace(float(np.percentile(k,99)),float(np.percentile(k,1)),y1-y0)
    for j,v in enumerate(vals):
        col=plt.cm.coolwarm(norm(v)); c=tuple(int(255*q) for q in col[:3])
        draw.line([(x0,y0+j),(x1,y0+j)],fill=c)
    draw.rectangle([x0,y0,x1,y1],outline=(60,60,60))
    kmin=float(np.percentile(k,1)); kmax=float(np.percentile(k,99))
    draw.text((x1+6,y1-4),f'{kmin:.3f}',fill=(30,30,30),font=f_tick,anchor='lm')
    draw.text((x1+6,y0-2),f'{kmax:.3f}',fill=(30,30,30),font=f_tick,anchor='lm')
    draw.text((x1+8,(y0+y1)//2),'signed\ncurvature\n(1/m)',fill=(30,30,30),font=f_tick,anchor='lm')
    p=D.ST if False else DST
    DST.mkdir(parents=True,exist_ok=True)
    im.save(DST/f't048_fixed_{rid}.png')
    print('wrote',DST/f't048_fixed_{rid}.png',im.size)

D=None
if __name__=='__main__':
    for rid in ['2024-1','2024-3','2024-10','2024-13','2024-16']:
        make(rid)
