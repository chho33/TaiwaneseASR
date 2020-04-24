export default {
  props: {
    filename  : { type: String, default: 'record'   },
    format    : { type: String, default: 'wav'      },
    headers   : { type: Object, default: () => ({}) },
    uploadUrl : { type: String, default: 'http://0.0.0.0:5000/translator/translateRaw' }
  }
}
