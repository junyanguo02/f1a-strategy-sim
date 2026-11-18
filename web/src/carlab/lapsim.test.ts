import {expect,it} from 'vitest'
import fixtures from '../../verification/carlab.json'
import bahrain from '../../public/data/carlab/tracks/2024-1.json'
import australia from '../../public/data/carlab/tracks/2024-3.json'
import spain from '../../public/data/carlab/tracks/2024-10.json'
import {applyWake,solveLap,type Car,type Wake} from './lapsim'
const tracks:Record<string,typeof bahrain>={'2024-1':bahrain,'2024-3':australia,'2024-10':spain}
const cases:{id:string;car:Car;wake:Wake|null;v:number[];total:number}[]=fixtures
for(const f of cases){
  it('matches Python '+f.id+(f.wake?' with wake':' in clean air'),()=>{
    const track=tracks[f.id]
    const r=solveLap(track.s,track.k,applyWake(f.car,f.wake??undefined))
    expect(r.v.length).toBe(f.v.length)
    const max=r.v.reduce((m,v,i)=>Math.max(m,Math.abs(v-f.v[i])),0)
    console.log(f.id,Boolean(f.wake),'s',Math.abs(r.total-f.total),'m/s',max)
    expect(Math.abs(r.total-f.total)).toBeLessThan(1e-6)
    expect(max).toBeLessThan(1e-6)
  })
}
