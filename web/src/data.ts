export type Compound = 'HARD'|'MEDIUM'|'SOFT'
export type Candidate = {label:string;compounds:Compound[];pit_laps:number[];mean_s:number;p10_s:number;p50_s:number;p90_s:number;delta_vs_best_s:number}
type Race = {race_id:string;year:number;round:number;name:string;laps:number;has_lab:boolean;has_timeline:boolean}
type Driver = {driver:string;team:string;color:string;finish_pos:number;line_style:'solid'|'dashed';gap:(number|null)[];gap_ref:(number|null)[];pos:number[];pit_laps:number[];lapped:boolean[]}
export type Timeline = {race_id:string;laps:number;drivers:Driver[];stints:{driver:string;start:number;end:number;compound:Compound;new:boolean}[];fcy:{type:'SC'|'VSC';start:number;end:number}[]}
export type Heat = {pit1:number[];pit2:number[];value:(number|null)[][];best:number[];sc_windows:number[][]}
export type Finding = {id:string;label:string;estimate:number;ci?:number[];n:number;source:string;se?:number;p?:number}
const obj = (v:unknown):v is Record<string,unknown> => v!==null && typeof v==='object' && !Array.isArray(v)
const num = (v:unknown):v is number => typeof v==='number' && Number.isFinite(v)
const int = (v:unknown):v is number => num(v) && Number.isInteger(v) && v>=1
const str = (v:unknown):v is string => typeof v==='string' && !!v.trim()
const list = (v:unknown,check:(v:unknown)=>boolean):v is unknown[] => Array.isArray(v)&&v.every(check)
const numbers = (v:unknown):v is number[] => Array.isArray(v)&&v.every(num)
const compound = (v:unknown)=>['SOFT','MEDIUM','HARD'].includes(String(v))
function check(ok:unknown, field:string):asserts ok {if(!ok) throw Error('数据合约不匹配 · '+field)}
async function json(name:string){
 const response=await fetch(import.meta.env.BASE_URL+'data/'+name+'.json')
 if(!response.ok) throw Error(name+' · HTTP '+response.status)
 const data:unknown=await response.json()
 check(obj(data)&&obj(data.units)&&str(data.source)&&str(data.generated_at),name)
 return data
}
export function validateTimeline(v:Record<string,unknown>, race:Race){
 const n=race.laps, lap=(x:unknown)=>int(x)&&x<=n
 const span=(x:Record<string,unknown>)=>int(x.start)&&int(x.end)&&x.start<=x.end&&x.end<=n
 check(v.race_id===race.race_id&&v.laps===n&&list(v.drivers,d=>obj(d)&&str(d.driver)&&str(d.team)&&str(d.color)&&int(d.finish_pos)&&['solid','dashed'].includes(String(d.line_style))&&[d.gap,d.gap_ref].every(a=>list(a,x=>num(x)||x===null)&&a.length===n)&&list(d.pos,int)&&d.pos.length===n&&list(d.pit_laps,lap)&&list(d.lapped,x=>typeof x==='boolean')&&d.lapped.length===n)&&v.drivers.length,'timeline.drivers')
 check(list(v.stints,s=>obj(s)&&str(s.driver)&&span(s)&&compound(s.compound)&&typeof s.new==='boolean')&&list(v.fcy,s=>obj(s)&&span(s)&&['SC','VSC'].includes(String(s.type))),'timeline.stints/fcy')
 return v as unknown as Timeline
}
export function moveStop(stops:number[],index:number,proposed:number,laps:number){
 const lower=index===0?1:stops[index-1]+1
 const upper=index===stops.length-1?laps-1:stops[index+1]-1
 return stops.map((v,i)=>i===index?Math.max(lower,Math.min(upper,Math.round(proposed))):v)
}
export async function loadData(){
 const [index,t,s,h,o,f]=await Promise.all(['mock_index','mock_timeline','mock_strategies','mock_heatmap','mock_outcomes','findings'].map(json))
 check(str(index.version)&&str(index.source_commit)&&list(index.races,r=>obj(r)&&str(r.race_id)&&int(r.year)&&int(r.round)&&str(r.name)&&int(r.laps)&&typeof r.has_lab==='boolean'&&typeof r.has_timeline==='boolean')&&index.races.length,'index')
 const race=index.races[0] as Race
 const timeline=validateTimeline(t,race)
 check(['prerace','posthoc'].includes(String(s.mode))&&int(s.n_sims)&&list(s.candidates,c=>obj(c)&&str(c.label)&&list(c.compounds,compound)&&list(c.pit_laps,int)&&c.compounds.length===c.pit_laps.length+1&&(c.pit_laps as number[]).every((v,i,a)=>v<race.laps&&(!i||v>a[i-1]))&&[c.mean_s,c.p10_s,c.p50_s,c.p90_s,c.delta_vs_best_s].every(num))&&s.candidates.length,'strategies')
 check(numbers(h.pit1)&&numbers(h.pit2)&&list(h.value,row=>list(row,x=>x===null||num(x))&&row.length===(h.pit1 as number[]).length)&&h.value.length===h.pit2.length&&numbers(h.best)&&h.best.length===2&&list(h.sc_windows,a=>numbers(a)&&a.length===2),'heatmap')
 check(numbers(o.paired_seconds)&&o.paired_seconds.length,'outcomes')
 check(list(f.findings,x=>obj(x)&&str(x.id)&&str(x.label)&&str(x.source)&&num(x.estimate)&&int(x.n)&&(x.ci===undefined||numbers(x.ci)&&x.ci.length===2))&&['green','adjusted','safety'].every(id=>(f.findings as Finding[]).some(x=>x.id===id)),'findings')
 return {race,timeline,candidates:s.candidates as Candidate[],heat:h as unknown as Heat,outcomes:o.paired_seconds,findings:f.findings as Finding[]}
}
export type Data = Awaited<ReturnType<typeof loadData>>
