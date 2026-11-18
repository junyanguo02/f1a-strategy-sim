import {useContext,useMemo,useState} from 'react'
import {Language,useText} from '../locale'
import {useResource} from '../resources'
import {applyWake,solveLap,type Car} from './lapsim'
import './carlab.css'

interface Circuit {id:string;names:string[];car:Car;fuel:number;lowFuel:number;highFuel:number;gamma:number;lo:number;hi:number;laps:number}
interface Params {circuits:Circuit[];wake:{model:string;a:number;mu_loss:number;g0_s:number}[]}
interface Track {id:string;s:number[];k:number[];x:number[];y:number[];mean:number[];low:number[];high:number[];delta:number;length:number;referenceTime:number}
interface Row {category:string;bin:string;strict_observed_pct:number;aero:number;grip:number;mixed:number}
type Mode='aero'|'grip'|'mixed'
const signed=(v:number)=> (v>=0?'+':'')+v.toFixed(3)
const colors=['#77aaff','#50c7cd','#ffc65d','#ff7664']
const formatPoints=(values:number[],x:(i:number)=>number,y:(v:number)=>number)=>values.map((v,i)=>x(i).toFixed(2)+','+y(v).toFixed(2)).join(' ')

function TrackChart({track,v}:{track:Track;v:Float64Array}){
  const t=useText()
  const xmin=Math.min(...track.x), ymax=Math.max(...track.y)
  const w=Math.max(...track.x)-xmin,h=ymax-Math.min(...track.y)
  const scale=Math.min(350/w,180/h)
  const x=(i:number)=>20+(350-w*scale)/2+(track.x[i]-xmin)*scale
  const y=(i:number)=>14+(180-h*scale)/2+(ymax-track.y[i])*scale
  const paths=['','','','']
  for(let i=0;i<v.length;i++){
    const j=(i+1)%v.length, group=[120,200,260].filter(c=>3.6*v[i]>=c).length
    paths[group]+='M'+x(i).toFixed(1)+' '+y(i).toFixed(1)+'L'+x(j).toFixed(1)+' '+y(j).toFixed(1)
  }
  return <section className="cl-card"><h2>{t('Circuit speed map','赛道速度图')}</h2>
    <svg viewBox="0 0 390 210" role="img" aria-label={t('Centreline coloured by simulated speed','按模拟车速着色的中心线')}>{paths.map((d,i)=><path key={i} d={d} fill="none" stroke={colors[i]} strokeWidth="4" strokeLinecap="round"/>)}</svg>
    <div className="cl-key">{['<120','120–200','200–260','≥260'].map((s,i)=><span key={s}><i style={{background:colors[i]}}/>{s}</span>)} km/h</div></section>
}

function SpeedChart({track,v}:{track:Track;v:Float64Array}){
  const t=useText(), n=v.length
  const shift=Math.round(track.delta/(track.s[1]-track.s[0]))
  const align=(a:number[])=>a.map((_,i)=>a[(i+shift+n)%n])
  const high=align(track.high),low=align(track.low),mean=align(track.mean),sim=Array.from(v,x=>x*3.6)
  const max=Math.ceil(Math.max(...sim,...high)/50)*50
  const x=(i:number)=>40+track.s[i]/track.length*546, y=(val:number)=>186-165*val/max
  const area=formatPoints(high,x,y)+' '+formatPoints(low,x,y).split(' ').reverse().join(' ')
  return <section className="cl-card"><h2>{t('Speed over distance','车速与距离')}</h2>
    <svg viewBox="0 0 610 220" role="img" aria-label={t('Simulated speed and five measured laps over distance','模拟车速与五个实测圈的车速对照')}>
    {[0,100,200,300].map(tick=><g key={tick}><path d={'M40 '+y(tick)+'h546'} stroke="#35414c"/><text x="32" y={y(tick)+4} textAnchor="end">{tick}</text></g>)}
    <polygon points={area} fill="#354755"/><polyline points={formatPoints(mean,x,y)} stroke="#c1d4df" fill="none" strokeWidth="1.8"/>
    <polyline points={formatPoints(sim,x,y)} stroke="#ff745f" fill="none" strokeWidth="2"/>
    <text x="40" y="207">0 km</text><text x="585" y="207" textAnchor="end">{(track.length/1000).toFixed(2)} km</text></svg>
    <div className="cl-key"><span><i style={{background:'#ff745f'}}/>{t('Simulation','模拟')}</span><span><i style={{background:'#c1d4df'}}/>{t('Measured mean','实测均值')}</span><span><i style={{background:'#354755'}}/>{t('Min–max','最小至最大')}</span> km/h</div>
    <small>{t('Timestamp-corrected reference, residual alignment','时间戳修正后的参考曲线，剩余对齐偏移')} {track.delta>=0?'+':''}{track.delta} m</small></section>
}

function Fuel({circuits}:{circuits:Circuit[]}){
  const t=useText(),lang=useContext(Language)
  return <section className="cl-card"><h2>{t('Physics check on fuel','燃油项物理核对')}</h2><p>{t('Five calibrated baselines. Lines show the 105–110 kg fuel range and the 90% posterior interval.','五站标定基线。线段分别表示 105–110 kg 的燃油范围和后验 90% 区间。')}</p>
   <div className="cl-key"><span><i style={{background:'#62cbd3'}}/>{t('Physics','物理值')}</span><span><i style={{background:'#ff9670'}}/>{t('Posterior γ','后验 γ')}</span> s/{t('lap','圈')}</div>
   {circuits.map(c=><div className="cl-fuel" key={c.id}><b>{c.names[lang==='en'?0:1]}</b><svg viewBox="0 0 185 25" aria-hidden="true">
   {[[c.lowFuel,c.highFuel,c.fuel,'#62cbd3'],[c.lo,c.hi,c.gamma,'#ff9670']].map(([lo,hi,m,color],i)=><g key={i} stroke={String(color)} fill={String(color)}><path d={'M'+(Number(lo)*2200)+' '+(6+i*12)+'H'+(Number(hi)*2200)} strokeWidth="3"/><circle cx={Number(m)*2200} cy={6+i*12} r="3"/></g>)}</svg><span>{c.fuel.toFixed(3)} / {c.gamma.toFixed(3)}</span></div>)}</section>
}

function Slider({name,value,min,max,step,unit,set}:{name:string;value:number;min:number;max:number;step:number;unit:string;set:(v:number)=>void}){
  return <label className="cl-slider"><span>{name} <b>{Number(value.toFixed(3))} {unit}</b></span><input type="range" min={min} max={max} step={step} value={value} onChange={e=>set(+e.target.value)}/></label>
}
function Experiment({track,c,params,rows}:{track:Track;c:Circuit;params:Params;rows:Row[]}){
  const t=useText(),[car,setCar]=useState(c.car),[gap,setGap]=useState(.6),[mode,setMode]=useState<Mode>('aero')
  const base=useMemo(()=>solveLap(track.s,track.k,c.car),[track,c])
  const {lap,slope}=useMemo(()=>{
    const solve=(m:number)=>solveLap(track.s,track.k,{...car,mass_kg:m})
    return {lap:solve(car.mass_kg),slope:(solve(car.mass_kg+5).total-solve(car.mass_kg-5).total)/10}
  },[track,car])
  const loss=useMemo(()=>{
    const w=params.wake.find(w=>w.model===mode)!
    return solveLap(track.s,track.k,applyWake(car,{gap,a:w.a,b:w.mu_loss,g0:w.g0_s})).total-lap.total
  },[params,track,mode,gap,car,lap])
  const update=(key:keyof Car,val:number)=>setCar(old=>({...old,[key]:val}))
  const bin=gap<.6?'0.3-0.6':gap<1?'0.6-1.0':gap<1.5?'1.0-1.5':'1.5-3.0'
  return <><div className="cl-stats">{[[t('Simulated lap','模拟圈时'),lap.total.toFixed(3)+' s'],[t('Change from baseline','相对标定基线'),signed(lap.total-base.total)+' s'],[t('Mass sensitivity','质量敏感度'),slope.toFixed(4)+' s/kg'],[t('Reference lap','参考圈时'),track.referenceTime.toFixed(3)+' s']].map(([k,v])=><div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
    <div className="cl-plots"><TrackChart track={track} v={lap.v}/><SpeedChart track={track} v={lap.v}/></div>
    <section className="cl-card"><div className="cl-top"><h2>{t('Car setup','车辆设定')}</h2><button onClick={()=>setCar(c.car)}>{t('Reset setup','恢复标定值')}</button></div>
    <div className="cl-controls"><Slider name={t('Fuel load','燃油质量')} value={car.mass_kg-798} min={0} max={110} step={.1} unit="kg" set={v=>update('mass_kg',798+v)}/>
    <Slider name={t('Wheel power','轮上功率')} value={car.power_kW} min={600} max={900} step={5} unit="kW" set={v=>update('power_kW',v)}/>
    <Slider name={t('Downforce area · ClA','下压力面积 · ClA')} value={car.cla_m2} min={1} max={8} step={.01} unit="m²" set={v=>update('cla_m2',v)}/>
    <Slider name={t('Tyre grip · μ','轮胎抓地 · μ')} value={car.mu} min={1} max={2.4} step={.01} unit="" set={v=>update('mu',v)}/></div>
    <small>{t('Dry mass 798 kg. Top speed identifies the ratio P/CdA; the power slider holds drag fixed.','干质量 798 kg。极速只能识别 P/CdA 比值；调节功率时保持阻力不变。')}</small></section>
    <div className="cl-bottom"><Fuel circuits={params.circuits}/><section className="cl-card"><h2>{t('Following another car','跟车损失')}</h2>
    <label>{t('Mechanism','机制')} <select value={mode} onChange={e=>setMode(e.target.value as Mode)}><option value="aero">{t('Downforce','下压力')}</option><option value="grip">{t('Tyre grip','轮胎抓地')}</option><option value="mixed">{t('Combined','两者组合')}</option></select></label>
    <Slider name={t('Gap','前车间距')} value={gap} min={.3} max={3} step={.1} unit="s" set={setGap}/>
    <strong>{t('Lap loss','每圈损失')} {signed(loss)} s</strong>
    <p>{t('Corner losses across five circuits at','五站弯速损失，间距档')} {bin} s{gap===3?t(' (adjacent bin)','（相邻分箱）'):''}</p>
    <table><thead><tr><th>{t('Corner','弯型')}</th><th>{t('Measured','实测')}</th><th>{t('T055 model','T055 模型')}</th></tr></thead><tbody>{['low','mid','high'].map((k,i)=>{
      const r=rows.find(x=>x.category===k&&x.bin===bin)!
      return <tr key={k}><th>{[t('Low speed','低速'),t('Medium speed','中速'),t('High speed','高速')][i]}</th><td>{r.strict_observed_pct.toFixed(2)}%</td><td>{r[mode].toFixed(2)}%</td></tr>})}</tbody></table>
    <small>{t('Measurements use a >3 s control. The models cannot reproduce the concentration of losses in slow corners. Its cause remains open.','实测对照组间距大于 3 秒。模型无法重现损失集中于低速弯的现象，原因仍是开放问题。')}</small></section></div></>
}
export default function CarLab(){
 const t=useText(),lang=useContext(Language),[id,setId]=useState('2024-1')
 const p=useResource<Params>('carlab/params'),d=useResource<{rows:Row[]}>('carlab/dirty_air'),g=useResource<Track>('carlab/tracks/'+id)
 const c=p.value?.circuits.find(c=>c.id===id)
 return <div className="cl"><header className="cl-head"><div><h1>{t('Car Lab','车辆实验室')}</h1><p>{t('A quasi-steady point-mass model rebuilt from the Racing Design and Experiment Workbench. Effective parameters cross-check measured speeds and fuel effects.','从早期「赛车设计与实验工作台」重写的准稳态质点模型。用有效参数交叉核对实测车速与燃油效应。')}</p></div>
 <label>{t('2024 circuit','2024 分站')} <select value={id} onChange={e=>setId(e.target.value)}>{p.value?.circuits.map(c=><option key={c.id} value={c.id}>{c.names[lang==='en'?0:1]}</option>)}</select></label></header>
 {p.value&&d.value&&g.value&&c?<Experiment key={id} track={g.value} c={c} params={p.value} rows={d.value.rows}/>:<p role="status">{p.error||d.error||g.error?t('Data unavailable','数据暂不可用'):t('Loading circuit…','正在读取赛道…')}</p>}</div>
}
