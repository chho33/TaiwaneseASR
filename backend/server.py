#!/usr/bin/env python
import configargparse
import onmt.opts as opts
from onmt.translate import TranslationServer, ServerModelError, ASRServer
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

STATUS_OK = "ok"
STATUS_ERROR = "error"


def start(config_file,
          url_root="./translator",
          host="0.0.0.0",
          port=5000,
          debug=True):

    def prefix_route(route_function, prefix='', mask='{0}{1}'):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)
        return newroute

    app = Flask(__name__)
    CORS(app)
    app.route = prefix_route(app.route, url_root)
    translation_server = ASRServer()
    translation_server.start(config_file)


    @app.route('/translate', methods=['POST'])
    def translate():
        inputs = list(map(lambda x: float(x), request.files.get("audio").read().decode().split(",")))
        inputs = [{"src": inputs}]
        out = {}
        try:
            translation, scores, n_best, times = translation_server.run(inputs)
            assert len(translation) == len(inputs)
            assert len(scores) == len(inputs)

            out = [[{"tgt": translation[i]}
                    for i in range(len(translation))]]
        except ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

        return jsonify(out)

    app.run(debug=debug, host=host, port=port, use_reloader=False,
            threaded=True)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description="OpenNMT-py REST Server")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    parser.add_argument("--url_root", type=str, default="/translator")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--config", "-c", type=str,
                       default="./available_models/asr.conf.json")

    args = parser.parse_args()
    start(args.config, url_root=args.url_root, host=args.ip, port=args.port,
          debug=args.debug)
