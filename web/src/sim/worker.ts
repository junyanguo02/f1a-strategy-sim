import type {Compound} from '../data'
import {commonDraws,simulate,type Draws} from './engine'
import {asStrategy,unpack,type RaceExport} from './inputs'
import {optimize} from './optimize'
export interface Init {kind:'init';config:RaceExport;sequences:Compound[][]}
let config:RaceExport
let sequences:Compound[][]

export type Scenario='baseline'|'green'|'hasSC'
export interface Request {id:number;scenario:Scenario;compounds:Compound[];pit_laps:number[]}
export interface Result {id:number;values:number[];baseline:string;samePlan:boolean}
function conditionalSC(D:Draws):Draws {
  const keep=D.status.map((row,s)=>Array.prototype.includes.call(row,2)?s:-1).filter(s=>s>=0)
  if(!keep.length)throw Error('NO_SC')
  return {seed:D.seed,n_sims:keep.length,d:keep.map(s=>D.d[s]),status:keep.map(s=>D.status[s]),
    phase:keep.map(s=>D.phase[s]),t_noise:keep.map(s=>D.t_noise[s]),
    u_stint:keep.map(s=>D.u_stint[s]),pit_u:keep.map(s=>D.pit_u[s])}
}
function prepare(scenario:Scenario){
  const {race,tyre}=unpack(config as RaceExport)
  if(scenario==='green'){
    race.lap1_prob={SC:0,VSC:0}
    race.hazard={SC:Array(5).fill(0),VSC:Array(5).fill(0)}
  }
  const raw=commonDraws(race,tyre,1000,config.seed)
  const draws=scenario==='hasSC'?conditionalSC(raw):raw
  let best:ReturnType<typeof optimize>|undefined
  for(const compounds of sequences){
    const result=optimize(race,tyre,compounds,draws)
    if(!best||result.mean<best.mean)best=result
  }
  if(!best)throw Error('缺少基准策略')
  const s=best.strategy
  const label=[s.start_compound,...s.stops.map(x=>x[1])].map(c=>c[0]).join('-')+' @'+s.stops.map(x=>x[0]).join(',')
  return {race,tyre,draws,label,total:simulate(race,tyre,s,draws)}
}
let previous:Scenario|undefined
let cached:ReturnType<typeof prepare>|undefined
self.onmessage=({data}:{data:Request|Init})=>{
  if('kind' in data){config=data.config;sequences=data.sequences;cached=undefined;return}
  try{
    if(!cached||data.scenario!==previous){cached=prepare(data.scenario);previous=data.scenario}
    const totals=simulate(cached.race,cached.tyre,asStrategy(data),cached.draws)
    const values=Array.from(totals,(v,i)=>v-cached!.total[i])
    const currentLabel=data.compounds.map(c=>c[0]).join('-')+' @'+data.pit_laps.join(',')
    self.postMessage({id:data.id,values,baseline:cached.label,samePlan:currentLabel===cached.label} satisfies Result)
  }catch(error){self.postMessage({id:data.id,error:String(error)})}
}
