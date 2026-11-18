import {it,expect} from 'vitest'
import reference from './python_repeats.json'
import {Random} from './random'
import {commonDraws,simulate,validate,type Race,type Tyre,type Strategy,type Draws} from './engine'
import {optimize} from './optimize'
import {asStrategy,unpack,type RaceExport} from './inputs'
import type {Candidate} from '../data'

const mean=(a:ArrayLike<number>)=>Array.from(a).reduce((s,v)=>s+v,0)/a.length
interface Fixture {race:Race;tyre:Tyre;strategies:Strategy[];totals:number[][];draws:Draws}
const fixtures=import.meta.glob<Fixture>('./__fixtures__/parity_*.json',{import:'default'})
const inputs=import.meta.glob<RaceExport>('../../public/data/races/*/sim_inputs_prerace.json',{import:'default'})
const candidates=import.meta.glob<{candidates:Candidate[]}>('../../public/data/races/*/strategies.json',{import:'default'})
for(const id of ['2024-1','2024-3','2024-10']){
  for(const mode of ['prerace','posthoc']){
    const load=fixtures['./__fixtures__/parity_'+id+'_'+mode+'.json']
    it('shared draws '+id+' '+mode,async()=>{
      const f=await load()
      expect(f.strategies).toHaveLength(3)
      let error=0
      f.strategies.forEach((s,j)=>{
        const got=simulate(f.race,f.tyre,s,f.draws)
        expect(got.length).toBe(f.totals[j].length)
        got.forEach((v,i)=>error=Math.max(error,Math.abs(v-f.totals[j][i])))
      })
      console.log('parity',id,mode,error)
      expect(error).toBeLessThan(1e-6)
    })
  }
  it('independent repeat confidence interval '+id,async()=>{
    const prefix='../../public/data/races/'+id+'/'
    const doc=await inputs[prefix+'sim_inputs_prerace.json']()
    const {race,tyre}=unpack(doc)
    const D=commonDraws(race,tyre,5000,doc.seed)
    const {candidates:list}=await candidates[prefix+'strategies.json']()
    const scores=list.map(c=>mean(simulate(race,tyre,asStrategy(c),D)))
    const winner=scores.indexOf(Math.min(...scores)), delta=scores[1]-scores[0]
    console.log('statistical',id,'Python',list[0].label,'TS',list[winner].label,'Python gap',list[1].delta_vs_best_s,'TS gap',delta)
    // Keep the single-run ranking and gap above as diagnostics. The sampling
    // unit for the CI is an independent run, not an individual lap or draw.
    const ref=reference.races.find(r=>r.id===id)!
    const names={S:'SOFT',M:'MEDIUM',H:'HARD'} as const
    // T041 reoptimised the live roster. Keep the historical control plans fixed.
    const fixed=ref.labels.map(label=>{
      const [order,stops]=label.split(' @')
      return {compounds:order.split('-').map(c=>names[c as keyof typeof names]),pit_laps:stops.split(',').map(Number)}
    })
    expect(doc.seed).toBe(ref.seed)
    expect(ref.means).toHaveLength(reference.repeats)
    const repeats:number[]=[]
    for(let run=0; run<reference.repeats; run++){
      const draws=commonDraws(race,tyre,reference.nSims,ref.seed+run)
      const [first,second]=fixed.map(p=>simulate(race,tyre,asStrategy(p),draws))
      repeats.push(second.reduce((s,v,i)=>s+(v-first[i]),0)/second.length)
    }
    const n=reference.repeats
    const varianceOfMean=(x:number[],m:number)=>x.reduce((s,v)=>s+(v-m)**2,0)/(n*(n-1))
    const ts=mean(repeats),py=mean(ref.means)
    const a=varianceOfMean(repeats,ts),b=varianceOfMean(ref.means,py)
    const df=(a+b)**2/((a*a+b*b)/(n-1))
    // With two groups of 20 runs, Welch df is in [19,38]. Rounding down
    // selects the conservative critical value from the SciPy t table.
    const critical=reference.t975[Math.max(19,Math.min(38,Math.floor(df)))-19]
    const margin=critical*Math.sqrt(a+b)
    const lo=ts-py-margin,hi=ts-py+margin
    console.log('repeat CI',id,{ts,py,difference:ts-py,df,lo,hi})
    expect(lo).toBeLessThanOrEqual(0)
    expect(hi).toBeGreaterThanOrEqual(0)
  },30000)
}
it('seeded random variates reproduce; Student(8) has mean 0 and variance 8/6',()=>{
  const a=new Random(704),b=new Random(704)
  for(let i=0;i<100;i++)expect(a.student(8)).toBe(b.student(8))
  const sample=Array.from({length:100000},()=>a.student(8))
  expect(Math.abs(mean(sample))).toBeLessThan(.02)
  expect(Math.abs(mean(sample.map(v=>v*v))-8/6)).toBeLessThan(.04)
})
it('validates dry compounds, integer, ordered and interior stops',()=>{
  const s=asStrategy({compounds:['HARD','MEDIUM'],pit_laps:[1]})
  expect(()=>validate(s,10)).not.toThrow()
  for(const lap of [0,1.5,10]) expect(()=>validate({...s,stops:[[lap,'MEDIUM']]},10)).toThrow()
  expect(()=>validate({...s,stops:[[3,'SOFT'],[3,'HARD']]},10)).toThrow()
  expect(()=>validate({...s,stops:[[9,'HARD']]},10)).toThrow()
})
it('grid mean agrees with direct simulation for every eligible one-stop plan',async()=>{
  const doc=await inputs['../../public/data/races/2024-10/sim_inputs_prerace.json']()
  const {race,tyre}=unpack(doc), D=commonDraws(race,tyre,50,36)
  const got=optimize(race,tyre,['SOFT','HARD'],D)
  const scores=[]
  for(let lap=8;lap<=race.total_laps-8;lap+=2){
    scores.push(mean(simulate(race,tyre,{start_compound:'SOFT',stops:[[lap,'HARD']]},D)))
  }
  expect(got.mean).toBeCloseTo(Math.min(...scores),8)
  expect(got.strategy.stops[0][0]).toBe(8+2*scores.indexOf(Math.min(...scores)))
})

it('rejects an impossible two-stop grid and under-sized common stint draws',async()=>{
  const f=await fixtures['./__fixtures__/parity_2024-10_prerace.json']()
  const race={...f.race,total_laps:20},D=commonDraws(race,f.tyre,5,2)
  expect(()=>optimize(race,f.tyre,['SOFT','HARD','MEDIUM'],D)).toThrow('可行')
  const plan=asStrategy({compounds:['HARD','MEDIUM'],pit_laps:[8]})
  expect(()=>simulate(race,f.tyre,plan,{...D,u_stint:D.u_stint.map(()=>[0])})).toThrow('段数')
})

it('two-stop grid uses the same shared-draw mean as direct enumeration',async()=>{
  const f=await fixtures['./__fixtures__/parity_2024-10_prerace.json']()
  const race={...f.race,total_laps:32}, D=commonDraws(race,f.tyre,75,17)
  const result=optimize(race,f.tyre,['SOFT','HARD','MEDIUM'],D)
  let min=Infinity
  for(let first=8;first<=16;first+=2){
    for(let second=first+8;second<=24;second+=2){
      min=Math.min(min,mean(simulate(race,f.tyre,{start_compound:'SOFT',stops:[[first,'HARD'],[second,'MEDIUM']]},D)))
    }
  }
  expect(result.mean).toBeCloseTo(min,8)
})
