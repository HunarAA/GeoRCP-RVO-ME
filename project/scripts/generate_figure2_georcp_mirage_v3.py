from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUTPUT_DIR = Path(r"E:\Papers\RVO_ME\paper_figures")
OUTPUT_BASENAME = "Figure2_GeoRCP_MIRAGE"
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "svg.fonttype": "none",
})

C = {
    "input":"#F5F5F5","preproc":"#EAF2F8","encoder":"#D6EAF8","global":"#D4E6F1",
    "local":"#E8F5E9","interaction":"#FCE8D5","region":"#FADBD8","curve":"#DDEBF7",
    "point":"#E2F0D9","output":"#FFF7F7","group":"#6F7F8F","border":"#404040",
    "arrow":"#4A4A4A","local_arrow":"#5E7E5D","text":"#202020"
}

def rounded_box(ax,x,y,w,h,text,fc,fontsize=12,ec=None,lw=1.35,radius=0.018,bold_first=False,z=3):
    ec=ec or C["border"]
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.010,rounding_size={radius}",
                     facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z)
    ax.add_patch(p)
    if bold_first and "\n" in text:
        first,rest=text.split("\n",1)
        ax.text(x+w/2,y+h*0.62,first,ha="center",va="center",fontsize=fontsize,fontweight="bold",color=C["text"],zorder=z+1)
        ax.text(x+w/2,y+h*0.31,rest,ha="center",va="center",fontsize=max(9,fontsize-1),color=C["text"],zorder=z+1)
    else:
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize,color=C["text"],zorder=z+1)
    return p

def group_box(ax,x,y,w,h,label,ec,label_color=None):
    lc=label_color or ec
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.018",
                     facecolor="none",edgecolor=ec,linewidth=1.25,linestyle=(0,(4,3)),zorder=1)
    ax.add_patch(p)
    ax.text(x+0.012,y+h+0.008,label,ha="left",va="bottom",fontsize=14,fontweight="bold",color=lc,zorder=7)
    return p

def line(ax,x1,y1,x2,y2,lw=1.25,color=None,z=4):
    ln=Line2D([x1,x2],[y1,y2],linewidth=lw,color=color or C["arrow"],solid_capstyle="round",zorder=z)
    ax.add_line(ln); return ln

def arrow(ax,x1,y1,x2,y2,lw=1.25,color=None,mutation=11,z=5):
    p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=mutation,
                      linewidth=lw,color=color or C["arrow"],shrinkA=0,shrinkB=0,
                      connectionstyle="arc3,rad=0",zorder=z)
    ax.add_patch(p); return p

def orth_arrow(ax,pts,lw=1.25,color=None,mutation=11,z=5):
    col=color or C["arrow"]
    for a,b in zip(pts[:-2],pts[1:-1]):
        line(ax,*a,*b,lw=lw,color=col,z=z)
    arrow(ax,*pts[-2],*pts[-1],lw=lw,color=col,mutation=mutation,z=z)

def label(ax,x,y,text,fs=10.5,color=None,style="normal",ha="center",va="center",weight="normal"):
    ax.text(x,y,text,fontsize=fs,color=color or C["text"],fontstyle=style,
            fontweight=weight,ha=ha,va=va,zorder=7)

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(18,11.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    # Input/preproc
    rounded_box(ax,0.025,0.855,0.105,0.070,"Input OCT\nB-scan",C["input"],fontsize=13,bold_first=True)
    rounded_box(ax,0.150,0.840,0.185,0.100,
                "Resolution-preserving\npreprocessing\n570×380 → 576×384\nsymmetric zero-padding",
                C["preproc"],fontsize=12,bold_first=True)
    arrow(ax,0.130,0.890,0.150,0.890)

    # groups
    group_box(ax,0.360,0.665,0.360,0.255,"Shared retinal representation","#5B84A6","#315C7D")
    group_box(ax,0.755,0.665,0.215,0.255,"Local spatial-detail pathway","#6A9A68","#446C43")

    rounded_box(ax,0.400,0.845,0.280,0.058,
                "MIRAGE-Base encoder\n12 Transformer blocks; full fine-tuning",
                C["encoder"],fontsize=13,bold_first=True)
    arrow(ax,0.335,0.890,0.400,0.874)

    # clean top-route from preprocessed input to LocalEncoder
    orth_arrow(ax,[(0.335,0.910),(0.350,0.910),(0.350,0.968),(0.862,0.968),(0.862,0.890)],lw=1.15,color=C["local_arrow"],mutation=10)
    label(ax,0.610,0.977,"same padded input",fs=9.8,color="#446C43")

    # block taps
    xs=[0.382,0.468,0.554,0.640]
    labs=["Block 3\nselected tokens","Block 6\nselected tokens","Block 9\nselected tokens","Block 12\nselected tokens"]
    for x,t in zip(xs,labs):
        rounded_box(ax,x,0.770,0.073,0.050,t,C["encoder"],fontsize=10.3,lw=1.05)
        arrow(ax,x+0.0365,0.845,x+0.0365,0.820,lw=1.05,mutation=9)

    # global fusion
    rounded_box(ax,0.430,0.690,0.220,0.055,
                "GlobalFusion\nproject + concatenate + fuse → 256-D shared tokens",
                C["global"],fontsize=11.2,bold_first=True)
    ybus=0.755
    line(ax,0.4185,ybus,0.6765,ybus,lw=1.05)
    for x in xs:
        line(ax,x+0.0365,0.770,x+0.0365,ybus,lw=1.05)
    arrow(ax,0.540,ybus,0.540,0.745,lw=1.05,mutation=9)

    # local encoder
    rounded_box(ax,0.795,0.835,0.135,0.055,"LocalEncoder",C["local"],fontsize=12.5)
    rounded_box(ax,0.795,0.765,0.135,0.050,"Stride-2 features\n32 channels",C["local"],fontsize=10.8)
    rounded_box(ax,0.795,0.695,0.135,0.050,"Stride-4 features\n64 channels",C["local"],fontsize=10.8)
    arrow(ax,0.8625,0.835,0.8625,0.815,lw=1.05,mutation=9)
    arrow(ax,0.8625,0.765,0.8625,0.745,lw=1.05,mutation=9)

    # interaction
    rounded_box(ax,0.330,0.565,0.410,0.075,
                "Geometry-conditioned TaskInteraction\nregion / curve / point adapters + task embeddings\n2 Transformer layers, 8 attention heads",
                C["interaction"],fontsize=12.5,bold_first=True,ec="#B56B2D")
    arrow(ax,0.540,0.690,0.540,0.640,lw=1.35,mutation=12)
    label(ax,0.760,0.603,
          "Task streams remain separate while\nexchanging contextual information.",
          fs=10.3,style="italic",ha="left")

    # task group
    group_box(ax,0.035,0.125,0.930,0.355,"Task-specific fusion and prediction","#657B91","#3F5870")

    # Two separate buses
    token_bus_y=0.465
    local_bus_y=0.440
    line(ax,0.175,token_bus_y,0.825,token_bus_y,lw=1.25,color=C["arrow"])
    arrow(ax,0.535,0.565,0.535,token_bus_y,lw=1.25,mutation=10)
    label(ax,0.945,token_bus_y+0.010,"task-token streams",fs=9.2,ha="right")

    # local bus fed from both s2/s4 through a right-side vertical lane
    lane_x=0.950
    line(ax,0.930,0.790,lane_x,0.790,lw=1.1,color=C["local_arrow"])
    line(ax,0.930,0.720,lane_x,0.720,lw=1.1,color=C["local_arrow"])
    line(ax,lane_x,0.790,lane_x,local_bus_y,lw=1.1,color=C["local_arrow"])
    line(ax,0.175,local_bus_y,lane_x,local_bus_y,lw=1.1,color=C["local_arrow"])
    label(ax,0.945,local_bus_y+0.008,"local s2 + s4",fs=9.2,color="#446C43",ha="right")

    centers=[0.185,0.500,0.815]
    task_defs=[
        (0.085,0.370,0.200,0.065,"Region TaskFusion\nreassemble task tokens + local s4/s2",C["region"],"#A94442"),
        (0.400,0.370,0.200,0.065,"Curve TaskFusion\nreassemble task tokens + local s4/s2",C["curve"],"#4B78A6"),
        (0.715,0.370,0.200,0.065,"Point TaskFusion\nreassemble task tokens + local s4/s2",C["point"],"#5B8A57")
    ]
    for cx,(x,y,w,h,t,fc,ec) in zip(centers,task_defs):
        rounded_box(ax,x,y,w,h,t,fc,fontsize=11.3,bold_first=True,ec=ec)
        # task tokens feed left-of-center, local features right-of-center
        arrow(ax,cx-0.025,token_bus_y,cx-0.025,y+h,lw=1.05,mutation=9)
        arrow(ax,cx+0.025,local_bus_y,cx+0.025,y+h,lw=1.05,color=C["local_arrow"],mutation=9)

    # heads with good vertical spacing
    head_y=0.275; head_h=0.055
    for x,t,fc,ec in [
        (0.110,"RegionHead",C["region"],"#A94442"),
        (0.425,"CurveHead",C["curve"],"#4B78A6"),
        (0.740,"PointHead",C["point"],"#5B8A57"),
    ]:
        rounded_box(ax,x,head_y,0.150,head_h,t,fc,fontsize=12.5,ec=ec)
    for cx in centers:
        arrow(ax,cx,0.370,cx,head_y+head_h,lw=1.15,mutation=10)

    # outputs with space between every box
    oy=0.135; oh=0.095
    outs=[
        (0.048,0.122,"Semantic logits\n3 channels\nbackground / SRF / IRF\n384×576","#A94442"),
        (0.183,0.122,"Boundary logits\n2 channels\nSRF / IRF\n384×576","#A94442"),
        (0.335,0.105,"Centerline logits\n2 channels\nELM / EZ\n384×576","#4B78A6"),
        (0.452,0.105,"Distance field\n2 channels\nELM / EZ\n384×576","#4B78A6"),
        (0.569,0.105,"Column presence\n2 channels\nELM / EZ\n2×576","#4B78A6"),
        (0.708,0.112,"HF heatmap logits\n1 channel\n192×288\n(stride 2)","#5B8A57"),
        (0.832,0.112,"HF offsets\n2 channels\n192×288\nsubpixel","#5B8A57")
    ]
    for x,w,t,ec in outs:
        rounded_box(ax,x,oy,w,oh,t,C["output"],fontsize=10.2,ec=ec,lw=1.2)

    # output buses
    # region
    line(ax,0.185,head_y,0.185,0.255,lw=1.05)
    line(ax,0.109,0.255,0.244,0.255,lw=1.05)
    arrow(ax,0.109,0.255,0.109,oy+oh,lw=1.0,mutation=8)
    arrow(ax,0.244,0.255,0.244,oy+oh,lw=1.0,mutation=8)
    # curve
    line(ax,0.500,head_y,0.500,0.255,lw=1.05)
    line(ax,0.3875,0.255,0.6215,0.255,lw=1.05)
    for x in [0.3875,0.5045,0.6215]:
        arrow(ax,x,0.255,x,oy+oh,lw=1.0,mutation=8)
    # point
    line(ax,0.815,head_y,0.815,0.255,lw=1.05)
    line(ax,0.764,0.255,0.888,0.255,lw=1.05)
    for x in [0.764,0.888]:
        arrow(ax,x,0.255,x,oy+oh,lw=1.0,mutation=8)

    # supervision note
    label(ax,0.500,0.082,
          "Geometry-specific supervision: region masks + boundaries  |  "
          "curve centerline + distance + column presence (y-distribution and ELM-above-EZ losses)  |  "
          "HF heatmap + subpixel offsets",
          fs=10.2,color="#3F3F3F")

    legend=[
        Line2D([0],[0],marker="s",linestyle="None",markersize=11,markerfacecolor=C["encoder"],markeredgecolor=C["border"],label="MIRAGE/shared"),
        Line2D([0],[0],marker="s",linestyle="None",markersize=11,markerfacecolor=C["local"],markeredgecolor=C["border"],label="local detail"),
        Line2D([0],[0],marker="s",linestyle="None",markersize=11,markerfacecolor=C["interaction"],markeredgecolor="#B56B2D",label="task interaction"),
        Line2D([0],[0],marker="s",linestyle="None",markersize=11,markerfacecolor=C["region"],markeredgecolor="#A94442",label="region stream"),
        Line2D([0],[0],marker="s",linestyle="None",markersize=11,markerfacecolor=C["curve"],markeredgecolor="#4B78A6",label="curve stream"),
        Line2D([0],[0],marker="s",linestyle="None",markersize=11,markerfacecolor=C["point"],markeredgecolor="#5B8A57",label="point stream"),
    ]
    fig.legend(handles=legend,loc="lower center",ncol=6,frameon=False,bbox_to_anchor=(0.5,0.012),
               fontsize=10.8,columnspacing=1.25,handletextpad=0.45)

    fig.subplots_adjust(left=0.015,right=0.990,top=0.965,bottom=0.080)
    png=OUTPUT_DIR/f"{OUTPUT_BASENAME}.png"; svg=OUTPUT_DIR/f"{OUTPUT_BASENAME}.svg"
    fig.savefig(png,dpi=DPI,bbox_inches="tight",facecolor="white")
    fig.savefig(svg,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(png); print(svg)

if __name__=="__main__":
    main()
