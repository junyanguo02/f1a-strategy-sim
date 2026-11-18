import {useEffect,useRef,useState} from 'react'
import {OutcomeDistribution,PitHeatmap,StrategyGantt,Num} from '../charts'
import type {Candidate,Heat} from '../data'
import type {Request,Result,Scenario} from './worker'
import {useResource} from '../resources'
import {useText} from '../locale'
import type {Init} from './worker'
import type {RaceExport} from './inputs'
import {raceName,type RaceIndex} from '../races'

type Roster={candidates:(Candidate & {mc_se_s?:number})[];near_tie?:boolean;near_tie_with?:string[]}
function useComparison(request: Omit<Request,'id'>,config:RaceExport,roster:Roster){
  const worker=useRef<Worker|null>(null), counter=useRef(0), last=useRef(-Infinity)
  const [result,setResult]=useState<Result|null>(null),[error,setError]=useState('')
  useEffect(()=>{
    const w=new Worker(new URL('./worker.ts',import.meta.url),{type:'module'})
    worker.current=w
    w.postMessage({kind:'init',config,sequences:roster.candidates.map(c=>c.compounds)} satisfies Init)
    w.onmessage=({data})=>{
      if(data.id!==counter.current)return
      if(data.error)setError(data.error)
      else setResult(data)
    }
    w.onerror=e=>setError(e.message)
    return ()=>{w.terminate();worker.current=null}
  },[config,roster])
  const {scenario,compounds,pit_laps}=request
  useEffect(()=>{
    const id=++counter.current
    setResult(null);setError('')
    const timer=setTimeout(()=>{
      last.current=performance.now()
      worker.current?.postMessage({id,scenario,compounds,pit_laps} satisfies Request)
    },Math.max(0,60-(performance.now()-last.current)))
    return ()=>clearTimeout(timer)
  },[scenario,compounds,pit_laps])
  return {result,error}
}
function Experiment({config,roster,heat}:{config:RaceExport;roster:Roster;heat:Heat}){
 const t=useText(),plans=roster.candidates
 const [index,select]=useState(0),[stops,move]=useState(plans[0].pit_laps)
 const [scenario,setScenario]=useState<Scenario>('baseline')
 const plan={...plans[index],pit_laps:stops}
 const {result,error}=useComparison({...plan,scenario},config,roster)
 return <div className="lab"><aside className="panel"><h2>{t('Strategy settings','策略设置')}</h2>
 <label>{t('Pre-race preset','赛前预设')}<select value={index} onChange={e=>{const i=Number(e.target.value);select(i);move(plans[i].pit_laps)}}>{plans.map((p,i)=><option key={p.label} value={i}>{p.label}</option>)}</select></label>
 <label>{t('Safety Car scenario','安全车情景')}<select value={scenario} onChange={e=>setScenario(e.target.value as Scenario)}>{[['baseline','Historical baseline','历史基线'],['green','All green','全程绿旗'],['hasSC','Draws with SC','仅含 SC 的抽样']].map(([v,en,zh])=><option value={v} key={v}>{t(en,zh)}</option>)}</select></label>
 {roster.near_tie&&<p className="tie">{t('Pre-race Python ranking · statistically tied','赛前 Python 排名 · 统计上并列')}<br/>{[plans[0].label,...roster.near_tie_with??[]].join(' / ')}</p>}
  {index===0&&plans[1]?.mc_se_s!==undefined&&<small>{t('Pre-race runner-up minus first','赛前次选减首选')} · {plans[1].label} · +{plans[1].delta_vs_best_s.toFixed(3)} s ({t('paired SE','配对标准误')} {plans[1].mc_se_s.toFixed(3)} s)</small>}
 {index>0&&plan.mc_se_s!==undefined&&<small>{t('Preset paired MC standard error','预设配对差的蒙特卡洛标准误')} · {plan.mc_se_s.toFixed(3)} s</small>}
 </aside><div><StrategyGantt plan={plan} laps={config.total_laps} onPlanChange={move}/>
 <div className="pair"><PitHeatmap heat={heat}/>{result?.samePlan?<section className="panel"><small>{t('Paired simulation','配对模拟')}</small><h2>{t('Current plan matches the reference','当前计划与基准相同')}</h2><p>{result.baseline} · {t('Every paired difference is zero. Move a pit stop or select another plan to compare.','每组配对差均为零。移动进站圈或选择其他方案即可比较。')}</p></section>:result?<OutcomeDistribution values={result.values} baseline={result.baseline}/>:<section className="panel" role="status">{error?(error.includes('NO_SC')?t('No Safety Car in these draws.','这批抽样没有安全车。'):t('Simulation failed','模拟失败')+' · '+error):t('Comparing paired draws…','正在重算配对差…')}</section>}</div>
 <p className="caption">{t('The heatmap and presets use the pre-race Python export. The histogram compares your plan with the browser grid reference under the selected inputs and scenario. Negative seconds favour your plan.','热图与预设来自赛前 Python 导出。直方图在所选模式、情景下与浏览器网格基准逐次比较。负值表示当前方案更快。')} {result&&<Num>n = {result.values.length}</Num>}</p>
 </div></div>
}
function EventLab({race,mode}:{race:string;mode:string}){
 const prefix='races/'+race+'/', t=useText()
 const input=useResource<RaceExport>(prefix+'sim_inputs_'+mode),plans=useResource<Roster>(prefix+'strategies'),heat=useResource<Heat>(prefix+'heatmap')
 if(input.value&&plans.value&&heat.value)return <Experiment config={input.value} roster={plans.value} heat={heat.value}/>
 return <p role="status">{input.error||plans.error||heat.error||t('Loading race data…','正在加载分站数据…')}</p>
}
export function Lab(){
 const t=useText(),idx=useResource<RaceIndex>('index')
 const [race,choose]=useState('2024-10'),[mode,setMode]=useState('prerace')
 return <><header className="intro"><small className="kicker">{t('SINGLE-CAR SIMULATION','单车模拟')}</small><h1>{t('Strategy Lab','策略实验室')}</h1>
 <p>{mode==='prerace'?t("Pre-race: last season's tyre model + history before race day",'赛前：上一赛季轮胎模型，加上比赛日前的历史数据。'):t('Post-race: tyre model fitted on this race; other inputs as pre-race','赛后：轮胎模型用本场数据拟合，其他输入沿用赛前口径。')}</p>
 <p className="notice">{t('Browser results are approximate. See the Python backtests for authoritative figures.','浏览器结果为近似，权威数字见 Python 回测。')}</p></header>
 <div className="toolbar"><label>{t('Grand Prix','分站')} <select value={race} onChange={e=>choose(e.target.value)}>{idx.value?.races.filter(r=>r.has_lab).map(r=><option value={r.race_id} key={r.race_id}>{raceName(r,t)}</option>)}</select></label>
 <div className="seg">{['prerace','posthoc'].map((m,i)=><button key={m} aria-pressed={mode===m} onClick={()=>setMode(m)}>{i?t('Post-race','赛后'):t('Pre-race','赛前')}</button>)}</div></div>
 {idx.error&&<p role="alert">{idx.error}</p>}<EventLab key={race+mode} race={race} mode={mode}/></>
}
