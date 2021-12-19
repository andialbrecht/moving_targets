# TODO: Is the last op not applied? Issue with modulo?
import glob
import subprocess
from itertools import islice
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import imageio
import numpy as np

from django.conf import settings


def get_renderer(project):
    renderers = {
        'stripes': StripesRenderer,
        'img23': Img23Renderer,
    }
    return renderers[project.renderer](project)


class ProcessingRenderer:

    def __init__(self, project, width=480, height=320):
        self.w = width
        self.h = height
        self.project = project
        self.num_ops = self._get_num_ops()

    def _get_num_ops(self):
        num = 0
        for func in dir(self):
            if not func.startswith('op'):
                continue
            if int(func[2:]) > num:
                num = int(func[2:])
        return num

    def _split_param(self, value):
        b_value = format(value, f'0{self.project.param_bits}b')
        length = round(self.project.param_bits / 2)
        left, right = b_value[:length], b_value[length:]
        return int(left, 2), int(right, 2)

    def scale(self, value, left, right, xp=0, fp=None):
        if fp is None:
            fp = self.project.param_max_value
        return int(np.interp(value, (xp, fp), (left, right)))

    def scale2(self, value, left, right, xp=0):
        return self.scale(
            value, left, right, xp,
            fp=2**(self.project.param_bits / 2) - 1)

    def render_pde(self, sequence):
        p = self.project
        pde = list(self.processing_globals())
        pde.extend(list(self.header()))
        for gene in sequence:
            b_gene = iter(format(gene, f'0{p.gene_bits}b'))
            op = (int(''.join(islice(b_gene, p.op_bits)), 2) % self.num_ops) + 1
            p1 = int(''.join(islice(b_gene, p.param_bits)), 2)
            p2 = int(''.join(islice(b_gene, p.param_bits)), 2)
            p3 = int(''.join(islice(b_gene, p.param_bits)), 2)
            p4 = int(''.join(islice(b_gene, p.param_bits)), 2)
            func = getattr(self, f'op{op}', self.noop)
            for line in func(p1, p2, p3, p4):
                if line:
                    pde.append(line)
        pde.extend(list(self.footer()))
        return '\n'.join(pde)

    def render_image(self, pde):
        path = Path(mkdtemp())
        fname = path / f'{path.name}.pde'
        with open(fname, 'w') as f:
            f.write(pde)
        p = subprocess.run(
            [settings.PJ_CMD, f'--sketch={path}', '--run'])
        if p.returncode != 0:
            for idx, line in enumerate(pde.splitlines()):
                print(f'{idx + 1:4}:  {line}')
            raise RuntimeError('Processing failed.')
        with open(path / 'output.jpg', 'rb') as f:
            data = f.read()
        rmtree(path)
        is_dead = self.check_dead(data)
        return data, is_dead

    def check_dead(self, data):
        """data is the raw image, returns bool."""
        return False

    def processing_globals(self):
        yield from ()

    def header(self):
        yield from (
            'void setup() {',
            f'  size({self.w}, {self.h});',
            '  noStroke();',
            '  noLoop();',
            '  background(255);',
            '}',
            '',
            'void draw() {',
        )

    def footer(self):
        yield from (
            '  save("output.jpg");',
            '  exit();',
            '}',
        )

    def noop(self, p1, p2, p3, p4):
        yield from ()


class StripesRenderer(ProcessingRenderer):

    def check_dead(self, data):
        # Image is dead, when almost white.
        im = imageio.imread(data)
        is_dead = np.mean(im) > 252
        return is_dead

    def processing_globals(self):
        yield from (
            'int op7_line = 0;',
            'float op7_line_perc = 0.0;',
        )

    def op1(self, p1, p2, p3, p4):
        """Draws horizontal line."""
        p1 = self.scale(p1, 0, 255)
        p2 = self.scale(p2, 0, self.h)
        p3 = self.scale(p3, 0, self.h)
        yield f'stroke({p1});'
        yield f'line(0, {p2}, {self.w}, {p3});'

    def op2(self, p1, p2, p3, p4):
        """Blurs the image."""
        p1 = self.scale(p1, 1, 10)
        yield f'filter(BLUR, {p1});'

    def op3(self, p1, p2, p3, p4):
        """Draw random vertical lines."""
        p1 = self.scale(p1, 0, self.w)
        p2 = self.scale(p2, p1, self.w)
        p3 = self.scale(p3, 76, 100)
        p4, p5 = self._split_param(p4)
        mv = 2**(self.project.param_bits / 2) - 1
        p4 = self.scale(p4, 0, 120, fp=mv)
        yield from (
            f'randomSeed({p5});',
            f'for (int i={p1}; i < {p2}; i++) {{',
            f'  if (random(100) > {p3}) {{',
            f'    stroke(int(random(0, {p4})));',
            f'    line(i, 0, i + int(random(-4, 4)), {self.h});',
            f'  }}',
            f'}}',
        )

    def op4(self, p1, p2, p3, p4):
        """Blends the image at half."""
        yield f'blend(0, 0, {self.w} / 2, {self.h}, {self.w} / 2, 0, {self.w} / 2, {self.h}, HARD_LIGHT);'

    def op5(self, p1, p2, p3, p4):
        """Applies DILATE filter."""
        yield 'filter(DILATE);'

    def op6(self, p1, p2, p3, p4):
        """Applies THRESHOLD filter."""
        p1 = self.scale(p1, 0, 100) / 100.0
        yield f'filter(THRESHOLD, {p1});'

    def op7(self, p1, p2, p3, p4):
        """Render random pixels below anything black."""
        # TODO: Requires global variables
        p2 = self.scale(p2, 60, 100)
        p3 = self.scale(p3, 0, 100)
        yield from (
            f'randomSeed({p1});',
            f'loadPixels();',
            f'op7_line = 0;',
            f'op7_line_perc = 0.0;',
            f'for (int i=0; i < ({self.w} * {self.h}); i++) {{',
            f'  if (i % {self.w} == 0) {{',
            f'    op7_line = op7_line + 1;',
            f'    op7_line_perc = (op7_line * 100.0) / {self.h};',
            f'  }}',
            f'  if (random({p2}) > op7_line_perc && pixels[i] != color(255)) {{',
            f'    int value = ({self.w} * {self.h}) - int(random(i + random({p3})));',
            f'    value = max(0, min(({self.w} * {self.h}) - 1, value));',
            f'    pixels[value] = pixels[i];',
            f'    pixels[i] = color(255);',
            f'  }}',
            f'}}',
            f'updatePixels();',
        )

    def op8(self, p1, p2, p3, p4):
        """A bezier curve."""
        p1a, p1b = self._split_param(p1)
        p1b = self.scale2(p1b, 0, 255)
        p2a, p2b = self._split_param(p2)
        p2a = self.scale2(p2a, 0, 255)
        p2b = self.scale2(p2b, 0, 255)
        p1 = self.scale(p1, 0, self.w)
        p2 = self.scale(p2, 0, self.w)
        p3 = self.scale(p3, 0, self.h)
        p4 = self.scale(p4, 0, self.h)
        yield from (
            f'randomSeed({p1a});',
            f'stroke(int(random({p1b})), int(random({p2a})), int(random({p2b})));',
            f'fill(255);',
            f'bezier(random({p1}), random({p3}), random({p2}), random({p4}), random({p1}), random({p3}), random({p2}), random({p4}));',
        )

    def op9(self, p1, p2, p3, p4):
        """Blends the image randomly."""
        p1a, p1b = self._split_param(p1)
        p2a, p2b = self._split_param(p2)
        p3a, p3b = self._split_param(p3)
        p4a, p4b = self._split_param(p4)
        p1a = self.scale2(p1a, 0, self.w)
        p1b = self.scale2(p1b, 0, self.h)
        p2a = self.scale2(p2a, 0, self.w)
        p2b = self.scale2(p2b, 0, self.h)
        p3a = self.scale2(p3a, 0, self.w)
        p3b = self.scale2(p3b, 0, self.h)
        p4a = self.scale2(p4a, 0, round(self.w / 2))
        p4b = self.scale2(p4b, 0, round(self.h / 2))
        np.random.seed(p4)
        modes = np.array([
            'BLEND', 'ADD', 'SUBTRACT', 'DARKEST', 'LIGHTEST',
            #'DIFFERENCE',
            'EXCLUSION', 'MULTIPLY', 'SCREEN',
            'OVERLAY', 'HARD_LIGHT', 'SOFT_LIGHT',
            'DODGE', 'BURN'])
        mode = np.random.choice(modes)
        yield f'blend({p1a}, {p1b}, {p3a}, {p3b}, {p2a}, {p2b}, {p4a}, {p4b}, {mode});'


class Img23Renderer(ProcessingRenderer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources = glob.glob(str(settings.IMG23_SOURCES / '*.jpg'))
        self.words = self._load_words()

    def _load_words_spiegel(self):
        import requests
        from bs4 import BeautifulSoup
        import nltk
        from nltk.tokenize import RegexpTokenizer
        response = requests.get('https://spiegel.de')
        soup = BeautifulSoup(response.text)
        text = ' '.join(x.text for x in soup.findAll('h2'))
        sw = nltk.corpus.stopwords.words('german')
        tokenizer = RegexpTokenizer("\w+")
        tokens = tokenizer.tokenize(text)
        words = []
        for word in tokens:
            if word.lower() not in sw:
                words.append(word)
        return words

    def _load_words(self):
        return ['verschwinden', 'gemeinsam', 'Tag', 'Zeit']

    def check_dead(self, data):
        # Image is dead, when almost black.
        im = imageio.imread(data)
        is_dead = np.mean(im) < 8
        return is_dead

    def processing_globals(self):
        yield from (
            'PImage img;',
            'PImage dest;',
            'PImage mem;',
            'int loc;',
            'int x;',
            'int y;',
            'float r;',
            'float g;',
            'float b;',
            IMG23_OP1,
            IMG23_OP2,
            IMG23_OP3,
            IMG23_OP4,
            IMG23_OP5,
        )

    def header(self):
        path = settings.IMG23_SOURCES / '20210523_080618.jpg'
        yield from (
            'void setup() {',
            f'  size({self.w}, {self.h}, P3D);',
            '  noStroke();',
            '  noLoop();',
            '  smooth();',
            '  background(0);',
            f'  img = loadImage("{path}");',
            '}',
            '',
            'void draw() {',
        )

    def op1(self, p1, p2, p3, p4):
        num = self.scale(p1, 1, 100)
        np.random.seed(p2)
        repetitions = []
        for i in range(num):
            seed = np.random.randint(1, 100000)
            repetitions.append(f'op1({seed});')
        yield from repetitions

    def op2(self, p1, p2, p3, p4):
        """Add cinema stripes."""
        p1 = self.scale(p1, 1, 15)
        p2 = self.scale(p2, 100, 255)
        yield from (
            f'op2({p1}, {p2}, {p3});',
        )

    def op3(self, p1, p2, p3, p4):
        """Darken the whole image to bring current work to back."""
        p1 = self.scale(p1, 0, 200)
        yield from (
            f'op3({p1});',
        )

    def op4(self, p1, p2, p3, p4):
        """Copies part of original image."""
        seed, scale = self._split_param(p1)
        alpha = self.scale(p2, 100, 255)
        x, y = self._split_param(p3)
        x = self.scale2(x, 0, self.w)
        y = self.scale2(y, 0, self.h)
        w, h = self._split_param(p4)
        w = self.scale2(w, 0, int(self.w / 2))
        h = self.scale2(h, 0, int(self.h / 2))
        max_items = w * h
        # max_items / n = max. n color changes within a box
        scale = self.scale2(scale, int(max_items / 4), max_items)
        yield f'op4({seed}, {scale}, {alpha}, {x}, {y}, {w}, {h});'

    def op5(self, p1, p2, p3, p4):
        """Draws randomly (walker)."""
        x = self.scale(p1, 0, self.w - 1)
        y = self.scale(p2, 0, self.h - 1)
        steps, stepsize = self._split_param(p3)
        steps = self.scale2(steps, 10, 300)
        stepsize = self.scale2(stepsize, 2, 50)
        seed = p4
        yield f'op5({x}, {y}, {steps}, {seed}, {stepsize});'

    def op6(self, p1, p2, p3, p4):
        """Blurs the image."""
        p1 = self.scale(p1, 1, 10)
        yield f'filter(BLUR, {p1});'

    def op7(self, p1, p2, p3, p4):
        """Erodes the image."""
        yield 'filter(ERODE);'

    def op8(self, p1, p2, p3, p4):
        """Posterize filter."""
        level = self.scale(p1, 2, 255)
        yield f'filter(POSTERIZE, {level});'

    # def op7(self, p1, p2, p3, p4):
    #     p1 = self.scale(p1, 4, 64)
    #     p2a, p2b = self._split_param(p2)
    #     p3a, p3b = self._split_param(p3)
    #     p4a, p4b = self._split_param(p4)
    #     p2a = self.scale2(p2a, 0, 255)
    #     p2b = self.scale2(p2b, 0, 255)
    #     p3a = self.scale2(p3a, 0, 255)
    #     p3b = self.scale2(p3b, 0, 255)
    #     p4a = self.scale2(p4a, 0, self.w)
    #     p4b = self.scale2(p4a, 0, self.h)
    #     word = np.random.choice(self.words)
    #     yield from (
    #         f'textSize({p1});',
    #         f'fill({p2a}, {p2b}, {p3a}, {p3b});',
    #         f'text("{word}", {p4a}, {p4b});',
    #     )


IMG23_OP1 = """
void op1(int seed) {
  randomSeed(seed);
  int w1 = int(random(width));
  int h1 = int(random(height));
  int w2 = int(random(img.width));
  int h2 = int(random(img.height));
  dest = createImage(w1, h1, ARGB);
  color firstCol = color(0, 0, 0);
  float locDir = random(100);
  for (int x=0; x<w1; x++) {
    for (int y=0; y<h1; y++) {
      int loc;
      if (locDir > 50) {
        loc = (int)map(x, 0, w1, 0, w2) + (int)map(y, 0, h1, h2 - (int)map(y, 0, h1, 0, 3), h2) * img.width;
      } else {
        loc = (int)map(x, 0, w1, w2 - (int)map(x, 0, w1, 0, 3), w2) + (int)map(y, 0, h1, 0, h2) * img.width;
      }
      float r = red(img.pixels[loc]);
      float g = green(img.pixels[loc]);
      float b = blue(img.pixels[loc]);
      if (x==0 && y==0) {
        firstCol = color(r, g, b);
      }
      float alpha = abs(brightness(firstCol) - brightness(color(r, g, b)));
      //println(alpha);
      alpha = alpha * (1- abs(map(x, 0, w1, -1, 1)));
      alpha = alpha * (1- abs(map(y, 0, h1, -1, 1)));
      dest.pixels[x + y*w1] = color(r, g, b, alpha);
    }
  }
  mem = dest.copy();
  if (random(100) > 80) {
    dest.blend(mem, 0, 0, w1, h1, 0, 0, mem.width, mem.height, DARKEST);
  }
  // This isn't very stable on different sizes (and too blue)
  //if (random(100) > 80) {
  //  float saveArea = random(100);
  //  dest.blend(img, (int)map(random(100), 0, 100, 0, w1), (int)map(random(100), 0, 100, 0, h1), w1, h1, (int)map(random(100), 0, 100, saveArea, w1-saveArea), (int)map(random(100), 0, 100, saveArea, h1-saveArea), img.width, img.height, LIGHTEST);
  //}
  image(dest, int(random(width + 100)) - 100, int(random(height + 100)) - 100);
}
"""

IMG23_OP2 = """
void op2(int height_perc, int alpha, int seed) {
  randomSeed(seed);
  color brightest = img.pixels[0];
  float r1 = red(brightest);
  float g1 = green(brightest);
  float b1 = blue(brightest);
  for (int i=0; i<(img.width * img.height); i++) {
     if (saturation(img.pixels[i]) < saturation(brightest) &&
         brightness(img.pixels[i]) > brightness(brightest)){
       float r2 = red(img.pixels[i]);
       float g2 = green(img.pixels[i]);
       float b2 = blue(img.pixels[i]);
       brightest = color(r1 * 0.3 + r2 * 0.7,
                         g1 * 0.3 + g2 * 0.7,
                         b1 * 0.3 + b2 * 0.7);
     }
  }
  int h = int(height * (height_perc / 100.0));
  noStroke();
  fill(brightest, alpha);
  quad(0, 0, width, 0,
       width, max(0, h + ((int)random(20) - 10)),
       0, max(0, h + ((int)random(20) - 10)));
  quad(0, height, width, height,
       width, height - (max(0, h + ((int)random(20) - 10))),
       0, height - (max(0, h + ((int)random(20) - 10))));
}
"""


IMG23_OP3 = """
void op3(int alpha) {
  noStroke();
  fill(0, 0, 0, alpha);
  rect(0, 0, width, height);
}
"""


IMG23_OP4 = """
void op4(int seed, int scale, int alpha, int x, int y, int w, int h) {
  randomSeed(seed);
  PImage tmp = createImage(w, h, ARGB);
  int startO = (int)random(img.pixels.length);
  color col = img.pixels[(int)random(img.pixels.length)];
  for (int i=0; i<(tmp.width * tmp.height); i++) {
    if ((i % scale) == 0) {
        startO = startO + 1;
        col = img.pixels[(int)random(img.pixels.length)];
    }
    //tmp.pixels[i] = img.pixels[startO];
    tmp.pixels[i] = col;
  }
  int[] mask = new int[tmp.pixels.length];
  for (int i=0; i<tmp.pixels.length; i++) {
      mask[i] = alpha;
  }
  tmp.mask(mask);
  tmp.filter(BLUR, 1);
  image(tmp, x, y);
}
"""

IMG23_OP5 = """
void op5(int x, int y, int steps, int seed, int stepSize) {
  randomSeed(seed);
  int orig_x = (int)random(0, img.width - 1);
  int orig_y = (int)random(0, img.height - 1);
  int z = (int)random(-50, 50);
  int next_x, next_y, next_z;
  for (int i=0; i<steps; i++) {
     next_z = (int)random(-50, 50);
     next_x = constrain(x + (int)random(stepSize * -1, stepSize + 1), 0, width - 1);
     next_y = constrain(y + (int)random(stepSize * -1, stepSize + 1), 0, height - 1);
     orig_x = constrain(orig_x + (int)random(-1, 2), 0, img.width - 1);
     orig_y = constrain(orig_y + (int)random(-1, 2), 0, img.height - 1);
     //pixels[x + y * width] = img.pixels[orig_x + orig_y * img.width];
     stroke(img.pixels[orig_x + orig_y * img.height]);
     strokeWeight(random(1, 4));
     line(x, y, z, next_x, next_y, next_z);
     x = next_x;
     y = next_y;
     z = next_z;
  }
}
"""