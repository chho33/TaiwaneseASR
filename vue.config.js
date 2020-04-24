module.exports = {
  devServer: {
      disableHostCheck: true,
      headers: { 'Access-Control-Allow-Origin': '*' },
      proxy: 'http://localhost:5000'
  }
}
