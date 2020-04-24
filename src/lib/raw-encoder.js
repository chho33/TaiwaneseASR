export default class {
  constructor (options) {
    this.bufferSize = options.bufferSize
    this.samples    = options.samples
  }

  finish () {
    const blob = new Blob([this.samples], {type: 'audio/raw'})

    return {
      id   : Date.now(),
      blob : blob,
      url  : URL.createObjectURL(blob)
    }
  }
}
