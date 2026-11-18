import {lazy,Suspense,useState,useEffect} from 'react'
import {createRoot} from 'react-dom/client'
import {HashRouter,NavLink,Routes,Route,Link,useLocation} from 'react-router-dom'
import type {Timeline,Finding} from './data'
import {RaceTimeline,ForestPlot,TyreChart,CountChart,type CountData,Num} from './charts'
const Writeup = lazy(() => import('./writeup/Writeup'))
const Lab=lazy(()=>import('./sim/Lab').then(m=>({default:m.Lab})))
import type {RaceExport} from './sim/inputs'
import {theme} from './theme'
import {Language,useText} from './locale'
import {useResource} from './resources'
import {raceName,type RaceIndex} from './races'
import './style.css'

for(const [key,value] of Object.entries(theme))document.documentElement.style.setProperty('--'+key,value)
function Status({error}:{error?:string}){
 const t=useText()
 return <p role={error?'alert':'status'}>{error||t('Loading data…','正在加载数据…')}</p>
}
function TimelinePicker({story=false}:{story?:boolean}){
 const t=useText(),idx=useResource<RaceIndex>('index')
 const [race,choose]=useState(story?'2024-3':'2024-10')
 const data=useResource<Timeline>('races/'+race+'/timeline')
 const events=idx.value?.races.filter(r=>r.has_timeline&&(!story||['2024-10','2024-3'].includes(r.race_id)))
 const current=events?.find(r=>r.race_id===race)
 return <><label className="toolbar">{t('Race replay','比赛回放')} <select value={race} onChange={e=>choose(e.target.value)}>{events?.map(r=><option key={r.race_id} value={r.race_id}>{raceName(r,t)}</option>)}</select></label>
 {data.value&&current?<RaceTimeline key={race} data={data.value} title={raceName(current,t)}/>:<Status error={data.error||idx.error}/>}</>
}
function Home(){
 const t=useText(),f=useResource<{findings:Finding[]}>('findings')
 const stats=[[t('Green-flag pairs','绿旗配对'),'green','n',t('Dry teammate comparisons','干地队友对照')],[t('Extra stop under green','绿旗多停一次'),'green','estimate',t('Mean finish-time difference','完赛用时差均值')],[t('Adjusted estimate','调整后系数'),'adjusted','estimate',t('Grid and opening pace controlled','控制发车位与首段配速')],[t('Safety Car estimate','安全车点估计'),'safety','estimate',t('95% interval crosses zero','95% 区间跨零')]]
 return <><div className="cover"><div className="hero"><small className="kicker">{t('RACE STRATEGY RESEARCH','赛车策略研究')}</small><h1>{t('Do teams stop','车队在绿旗下')}<br/><span>{t('too often','是不是进站太多了？')}</span>{t(' under green?','')}</h1><p>{t('Compare teammates in the same race to see what an extra pit stop costs.','把同场队友放在一起，看多一次进站究竟花了多少时间。')}</p><div className="actions"><Link className="primary" to="/story">{t('Read the research','阅读研究')} ↗</Link><Link to="/writeup">{t('Read the write-up','阅读文章')}</Link><Link to="/lab">{t('Open Strategy Lab','打开策略实验室')} →</Link></div></div>
 {f.value?<div className="tiles">{stats.map(([label,id,key,note])=>{const row=f.value!.findings.find(f=>f.id===id)!;return <article key={label}><small>{label}</small><strong><Num>{key==='n'?row.n:(row.estimate>0?'+':'')+row.estimate.toFixed(1)}</Num><em>{key==='n'?t('pairs','对'):'s'}</em></strong><span>{note}</span></article>})}</div>:<Status error={f.error}/>}</div>
 <TimelinePicker/><h2 className="sectionTitle">{t('Methods','研究方法')}</h2><div className="methods">{[
 [t('Teammate comparisons','队友对照'),t('Compare cars from the same team in the same race.','比较同场同队车手的完赛用时。')],
 [t('Tyre model','轮胎模型'),t('Estimate compound degradation with posterior uncertainty.','从逐圈用时估计配方衰减，并保留后验不确定性。')],
 [t('Strategy backtests','策略回测'),t('Compare plans on shared draws with pre-race or post-race inputs.','用共同抽样比较方案，并区分赛前与赛后输入。')]].map(([title,body])=><article key={title}><h3>{title}</h3><p>{body}</p></article>)}</div></>
}
function Counts(){const d=useResource<CountData>('findings/stop-counts');return d.value?<CountChart data={d.value}/>:<Status error={d.error}/>}
function Tyres(){const d=useResource<RaceExport>('races/2024-10/sim_inputs_posthoc');return d.value?<TyreChart data={d.value}/>:<Status error={d.error}/>}
function Evidence(){const d=useResource<{findings:Finding[]}>('findings');return d.value?<ForestPlot findings={d.value.findings}/>:<Status error={d.error}/>}
function Figure({step}:{step:number}){
 const t=useText()
 return [<Counts/>,<Tyres/>,<Evidence/>,<TimelinePicker story/>,<Link className="primary limitLink" to="/lab">{t('Try your own strategy','尝试自己的策略')} →</Link>][step]
}
function Story(){
 const t=useText(),[active,setActive]=useState(0)
 const [mobile,setMobile]=useState(()=>matchMedia('(max-width:720px)').matches)
 useEffect(()=>{const mq=matchMedia('(max-width:720px)'),changed=()=>setMobile(mq.matches);mq.addEventListener('change',changed);return ()=>mq.removeEventListener('change',changed)},[])
 useEffect(()=>{const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting)setActive(Number((e.target as HTMLElement).dataset.step))},{rootMargin:'-10% 0px -55% 0px'})
 document.querySelectorAll('[data-step]').forEach(el=>observer.observe(el));return ()=>observer.disconnect()},[])
 const chapters=[
 [t('The question','问题'),t('A simulator that kept saying “stop less”','模拟器为什么总建议少停？'),t('A stop costs about 22.8 s under green and 19.3 s under a Safety Car. We asked a simulator built from public lap data how often each driver should stop.','绿旗进站损失约 22.8 秒，安全车下约 19.3 秒。我们用公开圈速数据构建模拟器，逐车比较推荐停数。'),t('Pre-race recommendations matched actual stop counts 47% of the time. Among disagreements, the model wanted fewer stops in 84% of cases.','赛前推荐有 47% 与实际停数一致。在不一致的车次中，模型有 84% 建议少停。')],
 [t('The model','模型'),t('77,652 laps, one tyre model per season','77,652 圈，每个赛季一个轮胎模型'),t('A Bayesian model separates fuel burn (about 0.06 s per lap), tyre degradation (soft 0.066, medium 0.059, hard 0.044 s per lap) and dirty air.','贝叶斯模型区分燃油消耗（约 0.06 秒/圈）、轮胎衰减（软 0.066、中 0.059、硬 0.044 秒/圈）和尾流损失。'),t('The multi-car simulator adds traffic, overtaking and Safety Cars. Strategies share random draws. Before race day we use the previous season’s tyre posterior and pace from the last three races.','多车模拟加入车流、超车与安全车，用同一组抽样比较策略。赛前采用上一赛季的轮胎后验与前 3 场比赛的配速。')],
 [t('The evidence','证据'),t('Same car. Same race. One more stop.','同队同场，多停一次'),t('Across 82 dry teammate pairs, the driver stopping once more under green finished 16.4 s behind on average (95% CI 8.8–25.0 s). Excluding forced stops and adjusting for grid and first-stint pace gives +10.3 s (SE 3.6).','82 对干地队友中，绿旗下多停一次的一方平均慢 16.4 秒（95% 区间 8.8 至 25.0 秒）。排除被迫进站并控制发车位和首段配速后为 +10.3 秒（标准误 3.6）。'),t('Under a Safety Car the estimate is −11.6 s across 26 pairs. Its interval, −28.4 to +3.3 s, crosses zero.','安全车下 26 对的点估计为 −11.6 秒，区间 −28.4 至 +3.3 秒跨零。')],
 [t('Two races','两场比赛'),t('Where the model agrees with the teams','模型在哪些场次与车队一致？'),t('In Spain, high degradation favoured a soft–medium–soft two-stop in the post-race model. Pre-race inputs favoured one stop.','在西班牙，高衰减使赛后模型偏向软–中–软两停，赛前模型倾向一停。'),t('In Australia the model preferred one stop, while most of the top ten ran medium–hard–hard. Switch between the two races to inspect their stints.','在澳大利亚，模型倾向一停，前十名多数采用中–硬–硬两停。切换分站可查看每位车手的轮胎段。')],
 [t('Limits','局限'),t('What the comparison can show','这些对照能说明什么？'),t('Teams choose their stops. Public timing data cannot capture team orders, track-position defence or risk preferences. The optimiser searches one- and two-stop plans in dry races.','停数由车队决定。公开计时无法捕捉车队指令、位置防守和风险偏好。优化器在干地比赛中搜索一停和两停方案。'),t('Within-model checks of tyre cliffs, circuit Safety Car rates, traffic, undercuts, tyre management and stint differences did not explain the gap. Those checks cannot establish the causes in real races.','模型内检验了轮胎性能骤降、分赛道安全车概率、车流、提前进站、保胎与段间差异。这些检验未解释差距，实际比赛中的原因仍待验证。')]]
 return <><header className="intro"><small className="kicker">{t('RESEARCH STORY','研究叙事')}</small><h1>{t('The cost of one more stop','一次额外进站的代价')}</h1></header><div className="story"><div>{chapters.map(([label,title,...body],i)=><article className="step" key={i} data-step={i}><small className="kicker">0{i+1} / {label}</small><h2>{title}</h2>{body.map(p=><p key={p}>{p}</p>)}{mobile&&<Figure step={i}/>}</article>)}</div>{!mobile&&<aside><small>{chapters[active][0]}</small><Figure step={active}/></aside>}</div></>
}
function App(){
 const [lang,set]=useState<'en'|'zh'>(()=>localStorage.getItem('pitwall-language')==='zh'?'zh':'en'),location=useLocation()
 useEffect(()=>{document.documentElement.lang=lang==='en'?'en':'zh-CN';localStorage.setItem('pitwall-language',lang)},[lang])
 useEffect(()=>{window.scrollTo(0,0)},[location.pathname])
 return <Language.Provider value={lang}><header className="bar"><div className="shell"><Link className="brand" to="/"><i/>PITWALL</Link><nav>{[['/','Overview','总览'],['/story','Research','研究叙事'],['/lab','Strategy Lab','策略实验室'],['/writeup','Write-up','文章']].map(([to,en,zh])=><NavLink key={to} end to={to}>{lang==='en'?en:zh}</NavLink>)}</nav><div className="seg languages">{(['en','zh'] as const).map(l=><button key={l} lang={l} aria-pressed={lang===l} onClick={()=>set(l)}>{l==='en'?'EN':'中文'}</button>)}</div></div></header>
 <main className="shell"><Routes><Route path="/" element={<Home/>}/><Route path="/story" element={<Story/>}/><Route path="/lab" element={<Suspense fallback={<Status/>}><Lab/></Suspense>}/><Route path="/writeup" element={<Suspense fallback={<Status/>}><Writeup/></Suspense>}/><Route path="*" element={<Link to="/">{lang==='en'?'Overview':'总览'}</Link>}/></Routes></main>
 <footer className="shell">{lang==='en'?'Unofficial research project. This project is not associated with Formula 1 or any team.':'非官方研究项目，与一级方程式或任何车队均无关联。'}</footer></Language.Provider>
}
createRoot(document.getElementById('root')!).render(<HashRouter><App/></HashRouter>)
