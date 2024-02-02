from flask import Flask, render_template, url_for

app = Flask(__name__)


def get_static(dir, filename):
    return url_for('static', filename=f'{dir}/{filename}')


@app.route('/')
def main():
    return render_template(
        'index.html',
        css_style=get_static('css', 'style.css'),
        js_script=get_static('js', 'app.js'),
        favicon=get_static('images', 'favicon.png'),
        loading_image=get_static('images', 'loading.png')
    )


if __name__ == "__main__":
    app.run(debug=True)
