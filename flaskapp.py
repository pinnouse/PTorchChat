import cb

import json
from flask import Flask, render_template, request, make_response

app = Flask(__name__)


@app.route("/api", methods=['GET'])
def api() -> None:
    s = request.args.get('s')
    
    if s is None or len(s) <= 0:
        resp = make_response("Enter a valid query")
        resp.status_code = 400
        return resp

    read = cb.eval_sentence(str(request.args.get('s')))
    print(request.args, s, type(s))
    resp = make_response(json.dumps({
        'in_sentence': s,
        'out_sentence': read
    }))
    resp.headers['Content-Type'] = 'application/json'
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=80)
    app.run(debug=True)
