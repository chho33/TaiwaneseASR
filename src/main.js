import Vue from 'vue'
import './plugins/axios'
import App from './App.vue'
import vuetify from './plugins/vuetify';
//import VueAudioRecorder from 'vue-audio-recorder'

Vue.prototype.$eventBus = Vue.prototype.$eventBus || new Vue
Vue.config.productionTip = false
//Vue.use(VueAudioRecorder)

new Vue({
  vuetify,
  render: h => h(App)
}).$mount('#app')

