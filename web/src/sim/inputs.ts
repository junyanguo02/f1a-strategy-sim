import type {Candidate} from '../data'
import type {Race,Tyre,Strategy} from './engine'

interface EventParams {lap1_prob:number;hazard:number[];durations:number[];lap_mult:number}
export interface RaceExport {
  total_laps:number; base_lap_time_s:number; seed:number
  pit_loss:{green:number[];fcy:Race['pit_fcy']}
  fcy:Record<'SC'|'VSC',EventParams>; tyre_posterior:Tyre
}
export function unpack(data:RaceExport){
  function both<K extends keyof EventParams>(k:K){return {SC:data.fcy.SC[k],VSC:data.fcy.VSC[k]}}
  const race:Race={total_laps:data.total_laps,base_lap_time_s:data.base_lap_time_s,
    pit_normal:data.pit_loss.green,pit_fcy:data.pit_loss.fcy,lap1_prob:both('lap1_prob'),
    hazard:both('hazard'),durations:both('durations'),lap_mult:both('lap_mult')}
  return {race,tyre:data.tyre_posterior}
}
export function asStrategy(p:Pick<Candidate,'compounds'|'pit_laps'>):Strategy{
  return {start_compound:p.compounds[0],stops:p.pit_laps.map((lap,i)=>[lap,p.compounds[i+1]])}
}
