import {useState,useEffect} from 'react'
const cache=new Map<string,Promise<unknown>>()
export function resource<T>(path:string):Promise<T>{
 if(!cache.has(path)){
   const url=new URL(import.meta.env.BASE_URL+'data/'+path+'.json',document.baseURI)
   cache.set(path,fetch(url).then(r=>{if(!r.ok)throw Error(path+' · '+r.status);return r.json()}).catch(e=>{cache.delete(path);throw e}))
 }
 return cache.get(path)! as Promise<T>
}
export function useResource<T>(path:string){
 const [state,set]=useState<{path:string;value?:T;error?:string}>({path})
 useEffect(()=>{let live=true
   resource<T>(path).then(value=>{if(live)set({path,value})},e=>{if(live)set({path,error:String(e)})})
   return ()=>{live=false}
 },[path])
 return state.path===path?state:{path}
}
