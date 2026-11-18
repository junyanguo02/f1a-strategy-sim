import {createContext,useContext} from 'react'
export const Language=createContext<'en'|'zh'>('en')
export function useText(){
 const lang=useContext(Language)
 return (en:string,zh:string)=>lang==='en'?en:zh
}
