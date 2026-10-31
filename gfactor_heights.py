"""Run with Python or Origin embedded Python. Original ASC files remain untouched."""
from pathlib import Path
from datetime import datetime
import re, math, zipfile, traceback, os
from xml.sax.saxutils import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION = 'P1F_FIX_V7'

ROOT = Path(r"C:\HGE_Data")
RANGE_NM = (550.0, 800.0)
ACCUMULATIONS = 20.0
DYES = ('Achiral','RChiral','SChiral')
PATTERN = re.compile(r'^(Achiral|RChiral|SChiral)_(P[123])_([A-I])_(LH|RH)_QWP(45|135)_A0\.asc$',re.I)

def read(path):
    a=np.loadtxt(path)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<5 or not np.isfinite(a).all():
        raise ValueError('Expected >=5 rows of two finite numeric columns: '+str(path))
    if not (np.diff(a[:,0])>0).all():raise ValueError('Wavelength not strictly increasing: '+str(path))
    return a[:,0],(a[:,1]-a[:5,1].mean())/ACCUMULATIONS

# Detection smoothing only; reported heights use the original unsmoothed g values.
SMOOTH_SIGMA_NM = 2.0
MIN_LOBE_WIDTH_NM = 8.0
MIN_RELATIVE_AMPLITUDE = 0.15

def first_pair(x,g):
    ids=np.flatnonzero((x>=RANGE_NM[0])&(x<=RANGE_NM[1]))
    if len(ids)<10 or not np.isfinite(g[ids]).all():
        raise ValueError('Insufficient or undefined g data in 550-800 nm')
    xx=x[ids];yy=g[ids]
    dx=np.median(np.diff(xx))
    if not np.allclose(np.diff(xx),dx,rtol=.05,atol=1e-5):
        raise ValueError('Lobe detector requires approximately uniform wavelength spacing')
    radius=max(1,int(np.ceil(3*SMOOTH_SIGMA_NM/dx)))
    t=np.arange(-radius,radius+1)*dx
    kernel=np.exp(-.5*(t/SMOOTH_SIGMA_NM)**2);kernel/=kernel.sum()
    smooth=np.convolve(np.pad(yy,(radius,radius),mode='edge'),kernel,mode='valid')
    noise=1.4826*np.median(np.abs(np.diff(yy)-np.median(np.diff(yy))))/np.sqrt(2)
    threshold=max(MIN_RELATIVE_AMPLITUDE*np.max(np.abs(smooth)),5*noise,1e-6)
    signs=np.where(smooth>=0,1,-1)
    edges=np.r_[0,np.flatnonzero(np.diff(signs))+1,len(signs)]
    lobes=[]
    for start,end in zip(edges[:-1],edges[1:]):
        segment=smooth[start:end];sgn=signs[start]
        k=start+(np.argmax(segment) if sgn>0 else np.argmin(segment))
        if abs(smooth[k])<threshold or xx[end-1]-xx[start]<MIN_LOBE_WIDTH_NM:continue
        # Reject clipped lobes without an interior extremum.
        if k in (start,end-1):continue
        raw=start+(np.argmax(yy[start:end]) if sgn>0 else np.argmin(yy[start:end]))
        if sgn*yy[raw]<=0:continue
        lobes.append((sgn,raw,start,end))
    for left,right in zip(lobes[:-1],lobes[1:]):
        if left[0]==right[0]:continue
        lo=ids[left[1] if left[0]<0 else right[1]]
        hi=ids[left[1] if left[0]>0 else right[1]]
        return int(lo),int(hi)
    raise ValueError('No complete first substantial opposite-sign lobe pair below 800 nm; inspect spectrum')

# Dataset-specific windows checked against the measured spectra.
EXTREMUM_WINDOWS = {
    ('SChiral','P1','F','LH'): {'min': (745.0, 775.0)},
    ('RChiral','P2','A','LH'): {'min': (740.0, 770.0)},
    ('Achiral','P1','F','LH'): {'min': (745.0, 775.0)},
    ('Achiral','P3','D','RH'): {'max': (748.0, 764.0)},
    ('Achiral','P3','G','LH'): {'max': (668.0, 680.0)},
    ('RChiral','P3','D','LH'): {'max': (677.0, 687.0)},
    ('RChiral','P3','D','RH'): {'min': (669.0, 679.0)},
    ('RChiral','P3','G','LH'): {'min': (730.0, 770.0)},
    ('SChiral','P3','A','LH'): {'min': (735.0, 775.0)},
    ('SChiral','P3','C','LH'): {'min': (740.0, 780.0)},
}

def apply_windows(x,g,lo,hi,key):
    for kind,(start,end) in EXTREMUM_WINDOWS.get(key,{}).items():
        ids=np.flatnonzero((x>=start)&(x<=end)&np.isfinite(g))
        if not len(ids):raise ValueError('No valid data in reviewed '+kind+' window')
        idx=int(ids[np.argmin(g[ids]) if kind=='min' else np.argmax(g[ids])])
        if idx in (ids[0],ids[-1]):raise ValueError('Reviewed '+kind+' extremum lies at window edge; inspect window')
        if kind=='min':lo=idx
        else:hi=idx
    return lo,hi

def gcurve(p45,p135,key=None):
    x,u=read(p45);xx,v=read(p135)
    if len(x)!=len(xx) or not np.allclose(x,xx,rtol=0,atol=1e-5):raise ValueError('45/135 wavelength grids differ')
    if x[0]>RANGE_NM[0] or x[-1]<RANGE_NM[1]:raise ValueError('Spectrum does not cover 550-800 nm')
    den=u+v;g=np.full_like(u,np.nan)
    np.divide(2*(u-v),den,out=g,where=np.abs(den)>1e-12)
    mask=(x>=RANGE_NM[0])&(x<=RANGE_NM[1])
    if not np.isfinite(g[mask]).all():raise ValueError('Undefined g-factor inside 550-800 nm; check zero denominator')
    ids=np.flatnonzero(mask)
    if len(ids)<2:raise ValueError('Insufficient points in analysis range')
    if key is None:
        match=PATTERN.match(Path(p45).name)
        if match:
            dye,p,a,hand,_=match.groups()
            dye=next(d for d in DYES if d.lower()==dye.lower())
            key=(dye,p.upper(),a.upper(),hand.upper())
    lo,hi=first_pair(x,g)
    lo,hi=apply_windows(x,g,lo,hi,key)
    if key==('SChiral','P1','F','LH') and not 745<=x[lo]<=775:
        raise ValueError('P1F_FIX_V7: SChiral P1-F LH minimum must be within 745-775 nm')
    return x,g,lo,hi,float(g[hi]-g[lo])

def plot(path,title,curves,ratio):
    fig=plt.figure(figsize=(11,7));gs=fig.add_gridspec(1,2,width_ratios=[4,1.4])
    ax=fig.add_subplot(gs[0]);bar=fig.add_subplot(gs[1]);bar.axis('off')
    info=[];maxabs=0
    for hand,color in (('LH','#e32727'),('RH','#165ddd')):
        x,g,lo,hi,height=curves[hand];mask=(x>=550)&(x<=850)
        ax.plot(x[mask],g[mask],color=color,lw=2,label=hand)
        ax.scatter([x[lo],x[hi]],[g[lo],g[hi]],s=48,color=color,edgecolors='white',zorder=4)
        # Marker labels identify extrema; full coordinates appear in the side panel.
        for idx,label in ((hi,'max'),(lo,'min')):
            ax.annotate(hand+' '+label,(x[idx],g[idx]),xytext=(6,12 if label=='max' else -20),textcoords='offset points',color=color,fontsize=9)
        xpos=0.25 if hand=='LH' else 0.72
        # Separate height panel avoids arrows obscuring the measured spectra.
        maxabs=max(maxabs,float(np.max(np.abs(g[mask]))))
        info.append((hand,color,x,g,lo,hi,height,xpos))
    limit=max(1.5,maxabs*1.18)
    ax.set(xlim=(550,850),ylim=(-limit,limit),xlabel='Wavelength (nm)',ylabel='g-factor',title=title)
    ax.axvspan(800,850,color='0.9',zorder=0)
    ax.set_xticks(np.arange(550,851,50));ax.axhline(0,color='0.65',lw=.7);ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    # Height arrows use the same y scale as the spectra.
    bar.set_ylim(-limit,limit);bar.set_xlim(0,1)
    for hand,color,x,g,lo,hi,height,xpos in info:
        bar.annotate('',xy=(xpos,g[hi]),xytext=(xpos,g[lo]),arrowprops=dict(arrowstyle='<->',color=color,lw=2))
        bar.hlines([g[lo],g[hi]],xpos-.10,xpos+.10,colors=color,lw=1)
        bar.text(xpos,limit*.87,hand+' height\n'+f'{height:.6g}',color=color,ha='center',fontsize=10)
    lines=[]
    for hand,color,x,g,lo,hi,height,xpos in info:
        lines += [f'{hand} max: {x[hi]:.2f} nm, g={g[hi]:.6g}',f'{hand} min: {x[lo]:.2f} nm, g={g[lo]:.6g}']
    fig.text(.97,.01,VERSION,ha='right',fontsize=8,color='0.4')
    fig.text(.1,.035,'\n'.join(lines),fontsize=9,va='bottom')
    fig.suptitle(f'LH/RH height ratio = {ratio:.6g}     |     First lobe pair, 550-800 nm',y=.94,fontsize=12)
    fig.subplots_adjust(left=.09,right=.97,top=.84,bottom=.23,wspace=.12)
    fig.savefig(path,dpi=180);plt.close(fig)

# Write the three result sheets without an Excel dependency.
def save_excel(path,tables):
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    def textcell(ref,value):return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'+''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,4))+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name="{d}" sheetId="{i}" r:id="rId{i}"/>' for i,d in enumerate(DYES,1))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,4))+'</Relationships>')
        for i,d in enumerate(DYES,1):
            rows=['<row r="1">'+''.join(textcell(c+'1',v) for c,v in zip('ABCD',('Structure','LH height','RH height','LH/RH ratio')))+'</row>']
            for r,values in enumerate(tables[d],2):
                cells=[textcell('A'+str(r),values[0])]
                for c,v in zip('BCD',values[1:]):
                    if v is not None:cells.append(f'<c r="{c}{r}"><v>{v:.15g}</v></c>')
                rows.append(f'<row r="{r}">'+''.join(cells)+'</row>')
            z.writestr(f'xl/worksheets/sheet{i}.xml',f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="1" width="18" customWidth="1"/><col min="2" max="4" width="22" customWidth="1"/></cols><sheetData>'+''.join(rows)+'</sheetData><autoFilter ref="A1:D28"/></worksheet>')

def measurement_files(root):
    # Prune test directories before descending into them.
    for current,dirs,files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.lower().endswith('_0-90_test')]
        for name in files:
            yield Path(current)/name

def main(root=ROOT):
    print('Running',VERSION,'from',Path(__file__).resolve())
    root=Path(root)
    if not root.is_dir():raise ValueError('Data folder inaccessible: '+str(root))
    out=Path(__file__).resolve().parent/('Gfactor_Heights_'+VERSION+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    index={};errors=[]
    for path in measurement_files(root):
        if not path.is_file():continue
        m=PATTERN.match(path.name)
        if not m:continue
        dye,p,a,h,q=m.groups();key=(dye.lower(),p.upper(),a.upper(),h.upper(),int(q))
        index.setdefault(key,[]).append(path)
    tables={d:[] for d in DYES};count=0
    for dye in DYES:
        (out/dye).mkdir()
        for p in ('P1','P2','P3'):
            for a in 'ABCDEFGHI':
                name=p+'-'+a;curves={}
                try:
                    for hand in ('LH','RH'):
                        paths=[]
                        for q in (45,135):
                            key=(dye.lower(),p,a,hand,q);matches=index.get(key,[])
                            if len(matches)!=1:raise ValueError(f'{hand} QWP{q}: expected one ASC file, found {len(matches)}')
                            paths.append(matches[0])
                        curves[hand]=gcurve(*paths,key=(dye,p,a,hand))
                    lh,rh=curves['LH'][4],curves['RH'][4]
                    if rh<=1e-12:raise ValueError('RH height is zero; ratio undefined')
                    ratio=lh/rh
                    plot(out/dye/f'{dye}_{p}_{a}.png',f'{dye} / {p} / {a}',curves,ratio)
                    tables[dye].append([name,lh,rh,ratio]);count+=1
                    for hand,(x,g,lo,hi,height) in curves.items():
                        ids=np.flatnonzero((x>=550)&(x<=800))
                        if lo in (ids[0],ids[-1]) or hi in (ids[0],ids[-1]):errors.append(f'{dye} {name} {hand}: extremum at analysis boundary; inspect plot.')
                    print('Done:',dye,name)
                except (ValueError,OSError) as e:
                    tables[dye].append([name,None,None,None]);errors.append(f'{dye} {name}: {e}');print('Skipped:',dye,name,e)
    save_excel(out/'Gfactor_Height_Ratios.xlsx',tables)
    if errors:(out/'Errors.txt').write_text('\n'.join(errors),encoding='utf-8')
    print(f'Finished: {count}/81 plots. Results: {out}')
    return out

if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();print(error)
        try:(Path(__file__).parent/'Errors.txt').write_text(error,encoding='utf-8')
        except OSError:pass
        raise
