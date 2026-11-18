import {Random} from './random'
import type {Compound} from '../data'

export const compounds: Compound[] = ['HARD','MEDIUM','SOFT']
type Events<T> = Record<'SC'|'VSC',T>
export interface Race {
  total_laps: number; base_lap_time_s: number; pit_normal: number[]
  pit_fcy: Events<Record<number,number[]>>
  lap1_prob: Events<number>; hazard: Events<number[]>; durations: Events<number[]>; lap_mult: Events<number>
}
export interface Tyre {
  rem_mean:number; k0:number[][]; beta:number[][]; kappa:number[][]
  gamma:number[]; sigma:number[]; nu:number[]; sd_stint:number[]
}
export interface Strategy {start_compound:Compound; start_age?:number; stops:[number,Compound][]}
export interface Draws {
  seed:number; n_sims:number; d:number[]; status:ArrayLike<number>[]; phase:ArrayLike<number>[]
  t_noise:ArrayLike<number>[]; u_stint:ArrayLike<number>[]; pit_u:ArrayLike<number>[]
}

/** strategy.py:16–29. A stop is the in-lap, not the first lap on new tyres. */
export function validate(strategy:Strategy, laps:number) {
  const tyres = [strategy.start_compound,...strategy.stops.map(s=>s[1])]
  if (tyres.some(c=>!compounds.includes(c)) || new Set(tyres).size<2) throw Error('干地比赛至少使用两种有效配方')
  let previous = 0
  for (const [lap] of strategy.stops) {
    if (!Number.isInteger(lap) || lap<=previous || lap>=laps) throw Error('进站圈必须严格递增且位于比赛内')
    previous=lap
  }
}
const sorted=(a:number[])=>a.slice().sort((a,b)=>a-b)
function pick(pool:number[], u:number) {
  if (!pool?.length) throw Error('进站损失或持续圈数样本为空')
  return pool[Math.min(Math.floor(u*pool.length),pool.length-1)]
}

/** engine.py:12–55. Balanced posterior rows, independent FCY episodes,
 * Student-t lap noise and shared pit/stint variates. */
export function commonDraws(race:Race, tyre:Tyre, nSims:number, seed:number, maxStints=6):Draws {
  const N=race.total_laps, count=tyre.nu.length, rng=new Random(seed)
  if (!Number.isInteger(nSims)||nSims<1||!count||tyre.nu.some(v=>!(v>0))) throw Error('无效抽样规模或自由度')
  const remainder=nSims%count
  const d=Array.from({length:nSims-remainder},(_,i)=>i%count)
  const tail=Array.from({length:count},(_,i)=>i)
  rng.shuffle(tail); d.push(...tail.slice(0,remainder)); rng.shuffle(d)
  const durations={SC:sorted(race.durations.SC),VSC:sorted(race.durations.VSC)}
  const out:Draws={seed,n_sims:nSims,d,status:[],phase:[],t_noise:[],u_stint:[],pit_u:[]}
  for (const row of d) {
    const status=new Uint8Array(N), phase=new Uint16Array(N), noise=new Float64Array(N), pit=new Float64Array(N)
    let remaining=0, active=0, age=0
    for (let i=0;i<N;i++) {
      if (!remaining) {
        // np.linspace uses multiples of 1/5 (including 0.6000000000000001).
        let bin=0
        while (bin<4 && (i+1)/N >= (bin+1)*0.2) bin++
        const sc=i===0?race.lap1_prob.SC:race.hazard.SC[bin]
        const vsc=i===0?race.lap1_prob.VSC:race.hazard.VSC[bin]
        if (rng.uniform()<sc+vsc) {
          active=rng.uniform()<sc/Math.max(sc+vsc,1e-12)?2:1
          remaining=pick(durations[active===2?'SC':'VSC'],rng.uniform())
          age=0
        }
      }
      if (remaining>0) { status[i]=active; phase[i]=++age; remaining-- }
      noise[i]=rng.student(tyre.nu[row]); pit[i]=rng.uniform()
    }
    out.status.push(status);out.phase.push(phase);out.t_noise.push(noise);out.pit_u.push(pit)
    out.u_stint.push(Float64Array.from({length:maxStints},()=>rng.normal()))
  }
  return out
}

/** engine.py:57–62, 93–100. Prepare sorted pools once per evaluation. */
export function pitPicker(race:Race) {
  const pools=[sorted(race.pit_normal),...(['VSC','SC'] as const).flatMap(t=>[1,2,3].map(p=>sorted(race.pit_fcy[t][p])))]
  return (s:number,phase:number,u:number)=>pick(pools[s===0?0:(s-1)*3+Math.min(phase,3)],u)
}

/** engine.py:64–106. Returns the Python 'total' column, one entry per draw. */
export function simulate(race:Race, tyre:Tyre, strategy:Strategy, draws:Draws):Float64Array {
  const N=race.total_laps
  validate(strategy,N)
  if(strategy.stops.length>=draws.u_stint[0].length) throw Error('段数超过共同抽样配置')
  const pickPit=pitPicker(race), stopAt=new Map(strategy.stops)
  const C=new Uint8Array(N), A=new Float64Array(N), J=new Uint16Array(N), pit=new Uint8Array(N)
  let compound=strategy.start_compound, age=strategy.start_age??1, stint=0
  for(let i=0;i<N;i++) {
    C[i]=compounds.indexOf(compound);A[i]=age-10;J[i]=stint
    const next=stopAt.get(i+1)
    if(next){pit[i]=1;compound=next;age=1;stint++}else age++
  }
  return Float64Array.from(draws.d,(d,s)=>{
    let total=0
    for(let i=0;i<N;i++){
      const status=draws.status[s][i], c=C[i], a=A[i]
      let time=status ? race.base_lap_time_s*race.lap_mult[status===2?'SC':'VSC']
        : race.base_lap_time_s+tyre.k0[d][c]+tyre.gamma[d]*(N-i-1-tyre.rem_mean)+tyre.beta[d][c]*a+tyre.kappa[d][c]*a*a
          +tyre.sd_stint[d]*draws.u_stint[s][J[i]]+tyre.sigma[d]*draws.t_noise[s][i]
      if(pit[i]) time+=pickPit(status,draws.phase[s][i],draws.pit_u[s][i])
      total+=time
    }
    return total
  })
}
