import type {Compound} from '../data'
import {compounds,pitPicker,validate,type Race,type Tyre,type Draws,type Strategy} from './engine'

/** optimize.py:10–41, mean objective. On shared draws the grid mean is linear
 * in each tyre coefficient. Sum coefficients across simulations once, then
 * score legal stint lengths without materializing every candidate's S×N laps. */
export function optimize(race:Race, tyre:Tyre, sequence:Compound[], draws:Draws) {
  const N=race.total_laps, S=draws.n_sims
  validate({start_compound:sequence[0],stops:sequence.slice(1).map((c,i)=>[i+1,c])},N)
  if(sequence.length>draws.u_stint[0].length) throw Error('段数超过共同抽样配置')
  const vector=()=>new Float64Array(N)
  const base=vector(), pit=vector(), tint=sequence.map(vector)
  const offset=compounds.map(vector), beta=compounds.map(vector), quad=compounds.map(vector)
  const pick=pitPicker(race)
  for(let s=0;s<S;s++){
    const d=draws.d[s]
    for(let i=0;i<N;i++){
      const status=draws.status[s][i]
      pit[i]+=pick(status,draws.phase[s][i],draws.pit_u[s][i])
      if(status){base[i]+=race.base_lap_time_s*race.lap_mult[status===2?'SC':'VSC'];continue}
      base[i]+=race.base_lap_time_s+tyre.gamma[d]*(N-i-1-tyre.rem_mean)+tyre.sigma[d]*draws.t_noise[s][i]
      for(let j=0;j<sequence.length;j++) tint[j][i]+=tyre.sd_stint[d]*draws.u_stint[s][j]
      for(let c=0;c<3;c++){offset[c][i]+=tyre.k0[d][c];beta[c][i]+=tyre.beta[d][c];quad[c][i]+=tyre.kappa[d][c]}
    }
  }
  let best:Strategy|undefined, mean=Infinity
  function visit(laps:number[]){
    if(laps.length<sequence.length-1){
      const first=laps.length?laps[laps.length-1]+8:8
      for(let lap=first;lap<=N-8*(sequence.length-laps.length-1);lap+=2) visit([...laps,lap])
      return
    }
    let sum=0,start=0
    for(let j=0;j<sequence.length;j++){
      const end=laps[j]??N,c=compounds.indexOf(sequence[j])
      for(let i=start;i<end;i++){
        const a=i-start+1-10
        sum+=base[i]+offset[c][i]+beta[c][i]*a+quad[c][i]*a*a+tint[j][i]
      }
      if(end<N)sum+=pit[end-1]
      start=end
    }
    if(sum/S<mean){mean=sum/S;best={start_compound:sequence[0],stops:laps.map((l,i)=>[l,sequence[i+1]])}}
  }
  visit([])
  if(!best) throw Error('该配方顺序没有可行的进站圈组合')
  return {strategy:best as Strategy,mean}
}
