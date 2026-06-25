import { mount } from 'svelte'
import Play from './components/Play/Play.svelte'
import './app.css'

const app = mount(Play, { target: document.getElementById('app')! })
export default app
