import {expect,it} from 'vitest'
import index from '../public/data/index.json'
import counts from '../public/data/findings/stop-counts.json'
import type {Heat,Timeline,Candidate} from './data'
import type {RaceExport} from './sim/inputs'

const files=import.meta.glob<string>('../public/data/races/*/*.json',{query:'?raw',import:'default'})
async function read<T>(id:string,name:string):Promise<T>{
 return JSON.parse(await files['../public/data/races/'+id+'/'+name+'.json']())
}
it('resolves every selectable event to its own timeline and available lab inputs',async()=>{
 for(const race of index.races){
   if(race.has_timeline){
     const data=await read<Timeline>(race.race_id,'timeline')
     expect(data.race_id).toBe(race.race_id)
     expect(data.drivers.every(d=>d.gap_ref.length===race.laps)).toBe(true)
   }
   if(!race.has_lab)continue
   for(const mode of ['prerace','posthoc']){
     const data=await read<RaceExport>(race.race_id,'sim_inputs_'+mode)
     expect(data.total_laps).toBe(race.laps)
     expect(data.tyre_posterior.nu.length).toBeGreaterThan(0)
   }
   const heat=await read<Heat>(race.race_id,'heatmap')
   expect(heat.value.length).toBe(heat.pit1.length)
   expect(heat.value.every(row=>row.length===heat.pit2.length)).toBe(true)
   const plans=await read<{candidates:Candidate[];near_tie_with?:string[]}>(race.race_id,'strategies')
   expect(plans.candidates.length).toBeGreaterThan(0)
   for(const label of plans.near_tie_with??[])expect(plans.candidates.map(p=>p.label)).toContain(label)
 }
},60000)
it('reproduces the story headline from driver-race counts',()=>{
 const sum=(key:'agree'|'more'|'fewer')=>counts.rows.reduce((a,r)=>a+r[key],0)
 const mismatch=sum('fewer')+sum('more')
 expect(Math.round(100*sum('agree')/(sum('agree')+mismatch))).toBe(47)
 expect(Math.round(100*sum('fewer')/mismatch)).toBe(84)
})
