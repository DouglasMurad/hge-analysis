"""Run inside Origin's embedded Python. Edit SETTINGS first."""
from pathlib import Path
import re, csv, json
from datetime import datetime
import numpy as np

# Data and acquisition settings
ROOT = r"C:\HGE_Data"
DYES = ('Achiral', 'RChiral', 'SChiral')
GROUPS = ('P1','P2','P3')
GEOMETRIES = tuple('ABCDEFGHI')
ACQUISITION_CONFIRMED = True  # User confirmed 20 accumulations.
# All files must use comparable settings. Values below divide recorded counts.
# Subtract the first-five-point mean, then divide by accumulations.
DEFAULT_SCALE = 20.0
SCALE_BY_FILENAME = {}  # e.g. {'Achiral_BG_QWP0_A0.asc': 20.0}
WINDOW_NM = (None, None)  # set resonance window before interpreting extrema
DENOMINATOR_FLOOR = 1e-12  # absolute floor in scaled intensity / ratio units

SAMPLE = re.compile(r'^(Achiral|RChiral|SChiral)_(P[123])_([A-I])_(LH|RH)_QWP(0|45|90|135)_A0\.asc$', re.I)
GOLD = re.compile(r'^(Achiral|RChiral|SChiral)_BG_QWP(0|45|90|135)_A0\.asc$', re.I)

def read_spectrum(path):
    a = np.loadtxt(path)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Expected two finite numeric columns: '+str(path))
    if not (np.diff(a[:,0]) > 0).all():
        raise ValueError('Wavelength must strictly increase: '+str(path))
    scale = float(SCALE_BY_FILENAME.get(path.name, DEFAULT_SCALE))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid acquisition scale: '+path.name)
    if len(a) < 5:
        raise ValueError('Need at least five rows for dark-offset correction: '+str(path))
    return a[:,0], (a[:,1]-np.mean(a[:5,1]))/scale

def divide(a,b):
    result = np.full_like(a, np.nan, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b)>DENOMINATOR_FLOOR)
    np.divide(a,b,out=result,where=valid)
    return result

def signals(i0,i45,i135,g0,g45,g135):
    r0,r45,r135 = divide(i0,g0),divide(i45,g45),divide(i135,g135)
    return {'Reflectivity':r0,'Raw_difference':i45-i135,
            'Differential_reflectivity':r45-r135,
            'g_intensity':divide(2*(i45-i135),i45+i135),
            'g_reflectivity':divide(2*(r45-r135),r45+r135)}

def discover(root):
    found, ignored = {}, []
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() != '.asc': continue
        m = SAMPLE.match(p.name)
        if m:
            d,g,a,h,q = m.groups(); key=(d.lower(),g.upper(),a.upper(),h.upper(),int(q))
        else:
            m = GOLD.match(p.name)
            if not m: ignored.append(str(p)); continue
            d,q = m.groups(); key=(d.lower(),'BG',int(q))
        if key in found: raise ValueError('Duplicate measurement: '+str(p)+' and '+str(found[key]))
        found[key]=p
    return found,ignored

def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def letter(c):
    out=''; n=c+1
    while n: n,r=divmod(n-1,26); out=chr(65+r)+out
    return out

def build_table(x,gold,samples):
    data=[x]+[gold[q] for q in (0,45,90,135)]+[gold[45]-gold[135]]
    names=['Wavelength']+[f'BG_QWP{q}' for q in (0,45,90,135)]+['BG_CD']
    formula={5:'C-E'}
    for j,a in enumerate('ABCDEFGHI'):
        base=6+12*j
        lh,rh=samples[a]['LH'],samples[a]['RH']
        data += [lh[q] for q in (0,45,90,135)]+[lh[45]-lh[135]]
        data += [rh[q] for q in (0,45,90,135)]+[rh[45]-rh[135]]
        data += [divide(lh[0],gold[0]),divide(rh[0],gold[0])]
        names += [f'{a}_LH_QWP{q}' for q in (0,45,90,135)]+[f'{a}_LH_CD']
        names += [f'{a}_RH_QWP{q}' for q in (0,45,90,135)]+[f'{a}_RH_CD',f'{a}_LH_Reflectivity',f'{a}_RH_Reflectivity']
        formula[base+4]=f'{letter(base+1)}-{letter(base+3)}'
        formula[base+9]=f'{letter(base+6)}-{letter(base+8)}'
        formula[base+10]=f'{letter(base)}/B'; formula[base+11]=f'{letter(base+5)}/B'
    data += [divide(2*(gold[45]-gold[135]),gold[45]+gold[135])];names += ['BG_g_factor']
    formula[114]='2*(C-E)/(C+E)'
    for j,a in enumerate('ABCDEFGHI'):
        base=6+12*j
        for hand,offset in (('LH',0),('RH',5)):
            v=samples[a][hand]; c=len(data)
            data.append(divide(2*(v[45]-v[135]),v[45]+v[135]));names.append(f'{a}_{hand}_g_factor')
            u,w=letter(base+offset+1),letter(base+offset+3)
            formula[c]=f'2*({u}-{w})/({u}+{w})'
    assert len(data)==133 and letter(132)=='EC'
    return data,names,formula

def main():
    import originpro as op
    root=Path(ROOT)
    if not root.is_dir(): raise ValueError('Network data folder is inaccessible: '+ROOT)
    out=root/('HGE_V2_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    index,ignored=discover(root);issues=[];cache={}
    def load(key,x=None):
        if key not in index: raise ValueError('Missing '+str(key))
        path=index[key]
        if path not in cache: cache[path]=read_spectrum(path)
        xx,y=cache[path]
        if x is not None and (len(x)!=len(xx) or not np.allclose(x,xx,rtol=0,atol=1e-5)):
            raise ValueError('Wavelength mismatch: '+str(path))
        return xx,y
    def graph(sheet,columns,title,metric,x,data):
        gp=op.new_graph(template='line',lname=title);g=gp[0]
        colors=['#000000','#FF0000','#0000FF'] if len(columns)==3 else ['#000000','#FF0000']
        for c,color in zip(columns,colors):
            p=g.add_plot(sheet,coly=c,colx=0,type='line');p.color=color
            p.set_cmd('-w 1000')  # Origin: 500 units = 1 point; width 2.
        prefix,group,a=title.split(' ')[0].split('_')
        prefix={'RChiral':'Rchiral','SChiral':'Schiral'}.get(prefix,prefix)
        if metric=='CD':
            labels=[f'{prefix}_BG_CD-QWP(45-135)',f'{prefix}_{group}_{a}_CD-QWP(45-135)_LH',f'{prefix}_{group}_{a}_CD-QWP(45-135)_RH']
        elif metric=='g-factor':
            labels=[f'{prefix}_BG_g-Factor',f'{prefix}_{group}_{a}_g-Factor_LH',f'{prefix}_{group}_{a}_g-Factor_RH']
        else:
            labels=[f'Reflectivity_{prefix}_{group}_{a}_LH',f'Reflectivity_{prefix}_{group}_{a}_RH']
        g.label('Legend').text='\n'.join(r'\l(%d) %s'%(i+1,label) for i,label in enumerate(labels))
        g.axis('x').title='Wavelength(nm)'
        g.axis('y').title={'CD':'CD','g-factor':'g-Factor','Reflectivity':'Reflectivity' }[metric]
        g.rescale()
        g.set_xlim(600 if metric=='Reflectivity' else 550,850,50)
        if metric=='Reflectivity':g.set_ylim(0,1,0.2)
        elif metric=='CD':g.set_ylim(-150,150,30)
        else:g.set_ylim(-1.5,1.5,0.3)
    op.lt_exec('pe_cd path:="/";')
    runfolder='HGE_V2_'+datetime.now().strftime('%H%M%S')
    op.lt_exec(f'pe_mkdir folder:="{runfolder}" cd:=1;')
    for dye,folder in zip(DYES,('(1)_Achiral','(2)_R-Chiral','(3)_S-Chiral')):
        op.lt_exec(f'pe_mkdir folder:="{folder}" cd:=1;')
        for group in GROUPS:
            op.lt_exec(f'pe_mkdir folder:="{group}" cd:=1;')
            try:
                x,g0=load((dye.lower(),'BG',0));gold={0:g0}
                for q in (45,90,135):gold[q]=load((dye.lower(),'BG',q),x)[1]
            except ValueError as e:
                issues.append(dict(Set=dye+'_'+group,Issue=str(e)));op.lt_exec('pe_cd path:="..";');continue
            samples={};valid=[]
            for a in 'ABCDEFGHI':
                samples[a]={}; good=True
                for hand in ('LH','RH'):
                    samples[a][hand]={}
                    for q in (0,45,90,135):
                        try:y=load((dye.lower(),group,a,hand,q),x)[1]
                        except ValueError as e:
                            y=np.full_like(x,np.nan);good=False;issues.append(dict(Set=f'{dye}_{group}_{a}_{hand}',Issue=str(e)))
                        samples[a][hand][q]=y
                if good:valid.append(a)
            data,names,formulas=build_table(x,gold,samples)
            w=op.new_sheet('w',lname=dye+'_'+group);w.cols=133
            for c,(y,name) in enumerate(zip(data,names)):
                w.from_list(c,y.tolist(),lname=(name if c==0 else dye+'_'+group+'_'+name),units='nm' if c==0 else '')
                # Store formulas as comments; worksheet cells contain calculated values.
                w.set_label(c,formulas.get(c,'(source - mean(source[1:5]))/20' if c else 'Wavelength'),'C')
            w.cols_axis('x'+'y'*132)
            for j,a in enumerate('ABCDEFGHI'):
                if a not in valid:continue
                base=6+12*j;title=f'{dye}_{group}_{a}'
                for cols,metric in (([base+10,base+11],'Reflectivity'),([5,base+4,base+9],'CD'),([114,115+2*j,116+2*j],'g-factor')):
                    graph(w,cols,title+' '+metric,metric,x,data)
            print('Completed',dye,group,'valid geometries:',len(valid))
            op.lt_exec('pe_cd path:="..";')
        op.lt_exec('pe_cd path:="..";')
    if issues:
        (out/'Errors.txt').write_text('\n'.join(r['Set']+': '+r['Issue'] for r in issues),encoding='utf-8')
    if not op.save(str(out/'HGE_V2.opju')):raise RuntimeError('Save failed; save Origin project manually.')
    print('Finished:',out)

if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        error=traceback.format_exc()
        try:
            target=Path(ROOT)/('HGE_Error_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt')
            target.write_text(error,encoding='utf-8')
        except OSError:
            print('Could not write error file to data folder.')
        raise
