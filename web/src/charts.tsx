import {useState,type ReactNode} from 'react'
import {line,area,scaleLinear,extent,bin,quantileSorted} from 'd3'
import type {Timeline,Finding,Heat,Candidate} from './data'
import type {RaceExport} from './sim/inputs'
import {moveStop} from './data'
import {theme,tyres} from './theme'
import {useText} from './locale'
export const Num=({children}:{children:ReactNode})=><span className="num">{children}</span>
const sign=(v:number)=>(v>0?'+':'')+v.toFixed(1)
export function RaceTimeline({data,title}:{data:Timeline;title:string}){
 const t=useText(),[metric,set]=useState<'gap_ref'|'gap'>('gap_ref')
 const [all,expand]=useState(false)
 const drivers=all?data.drivers:data.drivers.slice(0,5)
 const x=scaleLinear([1,data.laps],[48,748]),[lo,hi]=extent(drivers.flatMap(d=>d[metric].filter(v=>v!==null)))
 const y=scaleLinear([Math.min(0,lo??0),hi??1],[145,16]).nice()
 const path=line<number|null>().defined(v=>v!==null).x((_,i)=>x(i+1)).y(v=>y(v!))
 return <section className="panel"><div className="chartHead"><div><small>{t('Race timeline','比赛时间线')}</small><h2>{title}</h2></div><div className="seg">{(['gap_ref','gap'] as const).map((k,i)=><button key={k} aria-pressed={metric===k} onClick={()=>set(k)}>{i?t('To leader','距领跑者'):t('To reference','距参考线')}</button>)}</div></div>
 <p className="caption">{metric==='gap_ref'?t('Gap to a constant pace based on the winner’s race time.','参考线按冠军全程平均圈速匀速前进。'):t('Gap to the leader on each lap.','各圈领跑者为零基准。')} {t('Seconds by lap · dots mark stops.','纵轴为秒，横轴为圈，圆点为进站。')}</p>
 <div className="chartScroll"><svg style={{maxHeight:'none'}} viewBox={'0 0 770 '+(185+18*drivers.length)} role="img" aria-label={t('Lap gaps, pit stops, tyre stints and neutralisations','逐圈时间差、进站点、轮胎段和安全车时段')}>
 {data.fcy.map((e,i)=><g key={i}><rect x={x(e.start)} width={x(e.end+1)-x(e.start)} y="7" height="139" fill={e.type==='SC'?'#F5C40025':'#7CC8FF25'}/><text x={x(e.start)} y="15">{e.type}</text></g>)}
 {y.ticks(4).map(v=><g key={v}><path d={'M48 '+y(v)+'H748'} className="gridline"/><text x="36" y={y(v)+4} textAnchor="end">{v}</text></g>)}
 {x.ticks(7).map(v=><text key={v} x={x(v)} y="166" textAnchor="middle">{v}</text>)}
 {drivers.map((d,i)=><g key={d.driver}><path d={path(d[metric])??''} fill="none" stroke={d.color} strokeWidth="2.5" strokeDasharray={d.line_style==='dashed'?'6 4':undefined}/>{d.pit_laps.map(p=>d[metric][p-1]!=null&&<circle key={p} cx={x(p)} cy={y(d[metric][p-1]!)} r="3" fill={theme.panel} stroke={d.color} strokeWidth="2"/>)}<text x="6" y={188+18*i}>{d.driver}</text>{data.stints.filter(s=>s.driver===d.driver).map((s,j)=><rect key={j} x={x(s.start)} y={181+18*i} width={Math.max(0,x(Math.min(data.laps,s.end+1))-x(s.start))} height="7" fill={tyres[s.compound]??theme.muted}/>)}</g>)}</svg></div>
 <div className="legend">{drivers.map(d=><span key={d.driver}><i style={{borderColor:d.color,borderTopStyle:d.line_style}}/>{d.driver}</span>)}<button onClick={()=>expand(!all)}>{all?t('Top five','仅看前五'):t('Full field','完整阵容')}</button></div></section>
}
export function ForestPlot({findings}:{findings:Finding[]}){
 const t=useText(),x=scaleLinear([-32,32],[22,518])
 return <section className="panel"><small>{t('Teammate comparisons','队友对照')}</small><h2>{t('Finish-time difference','完赛用时差')}</h2>{findings.map(f=>{
 const label=t(({green:'Extra green-flag stop',adjusted:'Adjusted estimate',safety:'Extra SC / VSC stop'} as Record<string,string>)[f.id],f.label)
 return <div key={f.id} className="forest"><div>{label}<b><Num>{sign(f.estimate)}</Num> s</b></div><svg viewBox="0 0 540 32" aria-label={label} role="img"><path d="M270 0v32" className="gridline"/>{f.ci&&<path d={'M'+x(f.ci[0])+' 15H'+x(f.ci[1])} stroke={theme.muted} strokeWidth="3"/>}<circle cx={x(f.estimate)} cy="15" r="6" fill={f.estimate>0?theme.loss:theme.gain}/></svg><small>{f.ci?<>95% CI <Num>{f.ci.map(sign).join(' ~ ')}</Num> s</>:<>SE <Num>{f.se}</Num> s</>}</small></div>})}<p className="caption">{t('Positive means slower. The SC / VSC interval crosses zero.','正值表示更慢。SC / VSC 的区间跨零。')}</p></section>
}
export function StrategyGantt({plan,laps,onPlanChange}:{plan:Candidate;laps:number;onPlanChange:(v:number[])=>void}){
 const t=useText(),ends=[0,...plan.pit_laps,laps]
 return <section className="panel"><small>{t('Pit strategy','进站计划')}</small><h2>{t('Adjust pit laps','调整进站圈')}</h2><div className="track">{plan.compounds.map((c,i)=><span key={i} style={{width:(ends[i+1]-ends[i])/laps*100+'%',background:tyres[c],color:c==='SOFT'?'white':theme.bg}}>{c[0]}</span>)}{plan.pit_laps.map((v,i)=><button key={i} className="handle" role="slider" aria-label={t('Pit stop ','进站 ')+(i+1)} aria-valuemin={i?plan.pit_laps[i-1]+1:1} aria-valuemax={(plan.pit_laps[i+1]??laps)-1} aria-valuenow={v} style={{left:v/laps*100+'%'}} onPointerDown={e=>e.currentTarget.setPointerCapture(e.pointerId)} onPointerMove={e=>{if(e.currentTarget.hasPointerCapture(e.pointerId)){const b=e.currentTarget.parentElement!.getBoundingClientRect();onPlanChange(moveStop(plan.pit_laps,i,(e.clientX-b.left)/b.width*laps,laps))}}} onKeyDown={e=>{const step=({ArrowLeft:-1,ArrowRight:1,Home:-laps,End:laps} as Record<string,number>)[e.key];if(step){e.preventDefault();onPlanChange(moveStop(plan.pit_laps,i,v+step,laps))}}}><Num>{v}</Num></button>)}</div><p className="caption">{t('Drag a stop or use the arrow keys.','拖动进站点，或使用左右方向键。')}</p></section>
}
export function PitHeatmap({heat}:{heat:Heat}){
 const t=useText(), dx=280/heat.pit1.length,dy=143/heat.pit2.length
 const color=scaleLinear<string>().domain([0,Math.max(...heat.value.flat().map(v=>v??0))]).range([theme.focus,'#20313D'])
 return <section className="panel"><small>{t('Pre-race Python export','赛前 Python 导出')}</small><h2>{t('Two-stop window','两停窗口')}</h2><svg viewBox="0 0 330 185" role="img" aria-label={t('Time loss by first and second pit lap','首停与次停圈数对应的时间损失')}>{heat.value.map((row,i)=>row.map((v,j)=>v!==null&&<rect key={i+'-'+j} x={34+i*dx} y={4+(heat.pit2.length-1-j)*dy} width={dx-1} height={dy-1} fill={color(v)}><title>{heat.pit1[i]} / {heat.pit2[j]} · +{v} s</title></rect>))}{heat.pit2.map((v,i)=>i%5===0&&<text key={v} x="6" y={14+(heat.pit2.length-1-i)*dy}>{v}</text>)}{heat.pit1.map((v,i)=>i%5===0&&<text key={v} x={33+i*dx} y="165">{v}</text>)}</svg><p className="caption">{t('First stop across, second stop up. Brighter cells are faster.','横轴为首停圈，纵轴为次停圈。越亮表示越快。')}</p></section>
}
export function OutcomeDistribution({values,baseline}:{values:number[];baseline:string}){
 const t=useText(),domain=extent(values) as [number,number]
 if(domain[0]===domain[1]){domain[0]-=.5;domain[1]+=.5}
 const bins=bin().domain(domain).thresholds(10)(values),x=scaleLinear(domain,[20,315]),y=scaleLinear([0,Math.max(...bins.map(b=>b.length))],[125,12])
 return <section className="panel"><small>{t('Paired simulation','配对模拟')}</small><h2>{t('Time vs reference','相对基准的用时差')}</h2><svg viewBox="0 0 335 150" role="img" aria-label={t('Distribution of paired time differences in seconds','配对时间差的分布，单位秒')}>{bins.map(b=><rect key={b.x0} x={x(b.x0!)} y={y(b.length)} width={Math.max(1,x(b.x1!)-x(b.x0!)-2)} height={125-y(b.length)} fill={b.x0!<0?theme.gain:theme.loss}/>)}{x.ticks(6).map(v=><text x={x(v)} y="144" key={v}>{v}</text>)}</svg><p className="caption">{t('Probability of beating','击败基准策略的概率')} {baseline} · <Num>{(100*values.filter(v=>v<0).length/values.length).toFixed(1)}%</Num></p></section>
}

export type CountData={rows:{year:number;agree:number;fewer:number;more:number}[];source:string}
export function CountChart({data}:{data:CountData}){
 const t=useText(),groups=['agree','fewer','more'] as const
 const labels=[t('Agreement','一致'),t('Fewer','模型少停'),t('More','模型多停')]
 return <section className="panel"><small>{t('Pre-race · multi-car','赛前 · 多车')}</small><h2>{t('Recommended vs actual stops','推荐与实际停数')}</h2><svg viewBox="0 0 500 164" role="img" aria-label={labels.join(' / ')}>{data.rows.map((r,i)=>{let x=58;return <g key={r.year}><text x="4" y={40+i*45}>{r.year}</text>{groups.map((g,j)=>{const w=r[g]/(r.agree+r.fewer+r.more)*430,left=x;x+=w;return <g key={g}><title>{r.year} · {labels[j]} · {r[g]}</title><rect x={left} y={20+i*45} width={w} height="29" fill={[theme.focus,theme.gain,theme.loss][j]}/><text style={{fill:w<18?theme.text:theme.bg}} x={w<18?left+w/2-2:left+w/2} y={(w<18?14:39)+i*45} textAnchor="middle">{r[g]}</text></g>})}</g>})}</svg><div className="legend">{labels.map((s,i)=><span key={s}><i style={{borderColor:[theme.focus,theme.gain,theme.loss][i]}}/>{s}</span>)}</div><p className="caption">{t('Labels count driver-races. Each bar sums to 100%.','数字为车次。每条合计为 100%。')}</p></section>
}
export function TyreChart({data}:{data:RaceExport}){
 const t=useText(),p=data.tyre_posterior
 const curves=(['HARD','MEDIUM','SOFT'] as const).map((c,k)=>({c,rows:Array.from({length:35},(_,i)=>{
 const a=i+1-10,v=p.beta.map((row,j)=>p.k0[j][k]+row[k]*a+p.kappa[j][k]*a*a).sort((x,y)=>x-y)
 return {age:i+1,lo:quantileSorted(v,.05)!,mid:quantileSorted(v,.5)!,hi:quantileSorted(v,.95)!}})}))
 const values=curves.flatMap(c=>c.rows.flatMap(p=>[p.lo,p.hi])),x=scaleLinear([1,35],[43,465]),y=scaleLinear(extent(values) as [number,number],[187,15]).nice()
 type Point=typeof curves[number]['rows'][number]
 return <section className="panel"><small>{t('Spain · post-race posterior','西班牙 · 赛后后验')}</small><h2>{t('Tyre effect by age','胎龄与轮胎用时')}</h2><svg viewBox="0 0 480 216" role="img" aria-label={t('Tyre model median and 90% posterior intervals','三种配方中位数与 90% 后验区间')}>{y.ticks(5).map(v=><g key={v}><path className="gridline" d={'M43 '+y(v)+'H465'}/><text x="5" y={y(v)}>{v}</text></g>)}{curves.map(({c,rows})=><g key={c}><path fill={tyres[c]} opacity=".18" d={area<Point>().x(p=>x(p.age)).y0(p=>y(p.lo)).y1(p=>y(p.hi))(rows)!}/><path stroke={tyres[c]} fill="none" strokeWidth="3" d={line<Point>().x(p=>x(p.age)).y(p=>y(p.mid))(rows)!}/></g>)}{x.ticks(6).map(v=><text x={x(v)} y="205" key={v}>{v}</text>)}</svg><p className="caption">{t('Tyre age (laps) × lap-time effect (s). White: hard · yellow: medium · red: soft. Bands cover 90% of posterior draws.','横轴为胎龄（圈），纵轴为轮胎用时项（秒）。白硬、黄中、红软；色带覆盖中央 90% 后验抽样。')}</p></section>
}
