"""Label placement shared by all map reports: every label is always shown.

A label goes next to its point if there is room; otherwise it is moved to the nearest free spot
around the point and joined to it by a thin leader line. If no free spot exists, it goes to the
least crowded candidate. Nothing is hidden.

LABEL_JS        JavaScript `placeLabels(...)` for the canvas maps (inserted into each page)
place_labels()  the same idea for the static matplotlib PNGs
"""
import math

LABEL_JS = r"""
function placeLabels(g,L,d,bg,ink,lead,opt){
 /* L: [{x,y,text,r,bold}] in canvas px, in priority order. Every label is drawn. */
 opt=opt||{};const fs=(opt.size||11)*d,pad=4*d,h=fs+5*d,W=g.canvas.width,H=g.canvas.height,boxes=[];
 const anchors=L.map(l=>[l.x,l.y,(l.r||0)+2*d]);
 const dirs=[[0,-1],[0,1],[1,0],[-1,0],[.7,-.7],[-.7,-.7],[.7,.7],[-.7,.7],[.38,-.92],[-.38,-.92],[.38,.92],[-.38,.92],[.92,-.38],[-.92,-.38],[.92,.38],[-.92,.38]];
 const steps=[0,8,18,32,50,72,98,130,170].map(s=>s*d);
 const ov=(a,b)=>Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
 const placed=[];
 L.forEach((l,li)=>{g.font=`${l.bold?600:400} ${fs}px system-ui,sans-serif`;const w=g.measureText(l.text).width+2*pad,r=(l.r||0)+3*d;let best=null,bestCost=Infinity;
  outer:for(const s of steps)for(const[dx,dy]of dirs){
   const cx=l.x+dx*(r+s+w/2*Math.abs(dx)),cy=l.y+dy*(r+s+h/2*Math.abs(dy))+(dx!==0&&dy===0?0:0);
   let b=[cx-w/2,cy-h/2,cx+w/2,cy+h/2];
   if(b[0]<2*d)b=[2*d,b[1],2*d+w,b[3]];if(b[2]>W-2*d)b=[W-2*d-w,b[1],W-2*d,b[3]];if(b[1]<2*d)b=[b[0],2*d,b[2],2*d+h];if(b[3]>H-2*d)b=[b[0],H-2*d-h,b[2],H-2*d];
   let cost=0;for(const o of boxes)cost+=ov(b,o);for(const[ax,ay,ar]of anchors){const qx=Math.max(b[0],Math.min(ax,b[2])),qy=Math.max(b[1],Math.min(ay,b[3]));if(Math.hypot(qx-ax,qy-ay)<ar)cost+=ar*ar*.5}
   cost+=s*.01;if(cost<bestCost){bestCost=cost;best=b}if(cost<s*.01+1e-6)break outer}
  boxes.push(best);placed.push([l,best])});
 g.save();placed.forEach(([l,b])=>{const qx=Math.max(b[0],Math.min(l.x,b[2])),qy=Math.max(b[1],Math.min(l.y,b[3])),dist=Math.hypot(qx-l.x,qy-l.y);
  if(dist>(l.r||0)+6*d){g.strokeStyle=lead;g.globalAlpha=.7;g.lineWidth=1*d;g.beginPath();g.moveTo(l.x,l.y);g.lineTo(qx,qy);g.stroke()}});
 g.textAlign="center";g.textBaseline="middle";
 placed.forEach(([l,b])=>{g.globalAlpha=.9;g.fillStyle=bg;g.fillRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);g.globalAlpha=1;g.font=`${l.bold?600:400} ${fs}px system-ui,sans-serif`;
  g.fillStyle=l.color||ink;g.fillText(l.text,(b[0]+b[2])/2,(b[1]+b[3])/2+.5*d)});g.restore()}
"""


def place_labels(ax, items, fontsize=7, color="#0b0b0b", lead="#8a8984"):
    """Static version for matplotlib. items: [(x, y, text)] in data units, in priority order."""
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    to_px = ax.transData.transform
    W, H = fig.canvas.get_width_height()
    pt = fig.dpi / 72
    h = (fontsize + 4) * pt
    boxes, anchors = [], [to_px((x, y)) for x, y, _ in items]
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (.7, .7), (-.7, .7), (.7, -.7), (-.7, -.7),
            (.38, .92), (-.38, .92), (.38, -.92), (-.38, -.92), (.92, .38), (-.92, .38), (.92, -.38), (-.92, -.38)]
    steps = [0, 6, 14, 26, 42, 62, 86, 116, 150]

    def ov(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    for (x, y, text), (px, py) in zip(items, anchors):
        w = (len(text) * 0.56 * fontsize + 6) * pt
        r = 5 * pt
        best, best_cost = None, math.inf
        for s in steps:
            done = False
            for dx, dy in dirs:
                cx = px + dx * (r + s * pt + w / 2 * abs(dx))
                cy = py + dy * (r + s * pt + h / 2 * abs(dy))
                b = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                cost = sum(ov(b, o) for o in boxes)
                for ax_, ay_ in anchors:
                    qx, qy = max(b[0], min(ax_, b[2])), max(b[1], min(ay_, b[3]))
                    if math.hypot(qx - ax_, qy - ay_) < 4 * pt:
                        cost += 8 * pt * pt
                cost += s * 0.01
                if cost < best_cost:
                    best_cost, best = cost, b
                if cost < s * 0.01 + 1e-6:
                    done = True
                    break
            if done:
                break
        boxes.append(best)
        cx, cy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
        tx, ty = inv.transform((cx, cy))
        qx, qy = max(best[0], min(px, best[2])), max(best[1], min(py, best[3]))
        if math.hypot(qx - px, qy - py) > 7 * pt:
            ex, ey = inv.transform((qx, qy))
            ax.plot([x, ex], [y, ey], color=lead, lw=0.5, alpha=0.8, zorder=4)
        ax.text(tx, ty, text, fontsize=fontsize, ha="center", va="center", color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="#fcfcfb", ec="none", alpha=0.85))
