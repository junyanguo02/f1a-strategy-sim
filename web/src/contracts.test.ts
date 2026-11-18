import {describe,it,expect,vi,afterEach} from 'vitest'
import {loadData,moveStop} from './data'
const files=import.meta.glob(['../public/data/mock_*.json','../public/data/findings.json'],{eager:true,import:'default'})
function serve(change:(name:string,data:any)=>void=()=>{}){
 vi.stubGlobal('fetch',async(url:string)=>{const name=url.split('/').pop(),data=structuredClone(files['../public/data/'+name]);change(name!,data);return new Response(JSON.stringify(data))})
}
afterEach(()=>vi.unstubAllGlobals())
describe('data contracts',()=>{
 it('loads and validates the packaged fixtures',async()=>{serve();const d=await loadData();expect(d.timeline.laps).toBe(d.race.laps);expect(d.findings).toHaveLength(3)})
 it('rejects truncated reference gaps',async()=>{serve((n,d)=>{if(n==='mock_timeline.json')d.drivers[0].gap_ref.pop()});await expect(loadData()).rejects.toThrow('timeline')})
 it('rejects ragged heatmap rows',async()=>{serve((n,d)=>{if(n==='mock_heatmap.json')d.value[0].pop()});await expect(loadData()).rejects.toThrow('heatmap')})
 it('reports missing files',async()=>{vi.stubGlobal('fetch',async()=>new Response(null,{status:404}));await expect(loadData()).rejects.toThrow('404')})
})
describe('pit lap edits',()=>{
 it.each([[0,-4,[1,40]],[0,50,[39,40]],[1,7,[20,21]],[1,100,[20,65]],[1,31.6,[20,32]]])('bounds stop %i at %i',(index,lap,expected)=>{expect(moveStop([20,40],Number(index),Number(lap),66)).toEqual(expected)})
})

describe('browser simulator input fixture',()=>{
 it('keeps the seven tyre parameters aligned for all 200 synthetic draws',()=>{
   const fixture=files['../public/data/mock_sim_inputs.json'] as {tyre:Record<string,unknown[]>;race:{total_laps:number}}
   for(const key of ['k0','beta','kappa','sigma','nu','gamma','sd_stint']){
     const draws=fixture.tyre[key]
     expect(draws).toHaveLength(200)
     for(const value of draws){
       if(['k0','beta','kappa'].includes(key)){
         expect(Array.isArray(value) && value.length===3 && value.every(Number.isFinite)).toBe(true)
       }else expect(Number.isFinite(value)).toBe(true)
     }
   }
   expect(fixture.race.total_laps).toBe(66)
 })
})
