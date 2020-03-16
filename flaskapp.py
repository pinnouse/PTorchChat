import cb

import json
from flask import Flask, render_template, request, make_response

app = Flask(__name__)


@app.route("/api")
def api() -> None:
    read = cb.eval_sentence(str(request.args.get('s')))
    resp = make_response(json.dumps({
        'in_sentence': str(request.args.get('s')),
        'out_sentence': read
    }))
    resp.headers['Content-Type'] = 'application/json'
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
