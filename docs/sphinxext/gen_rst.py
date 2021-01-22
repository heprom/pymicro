"""
Example generation for the scikit learn

Generate the rst files for the examples by iterating over the python
example files.

Files that generate images should start with 'plot'

"""
import os
import sys
import shutil
import traceback
import glob

import token, tokenize

rst_template = """

:orphan:

.. _example_%(short_fname)s:

%(docstring)s

**Pythonnn source code:** :download:`%(fname)s <%(fname)s>`

.. literalinclude:: %(fname)s
   :lines: %(end_row)s-
   """

plot_rst_template = """

:orphan:

.. _example_%(short_fname)s:

%(docstring)s

%(image_list)s

**Python source code:** :download:`%(fname)s <%(fname)s>`

.. literalinclude:: %(fname)s
   :lines: %(end_row)s-
   """

# The following strings are used when we have several pictures: we use
# an html div tag that our CSS uses to turn the lists into horizontal
# lists.
HLIST_HEADER = """
.. rst-class:: horizontal

"""

HLIST_IMAGE_TEMPLATE = """
    *

      .. image:: images/%s
            :scale: 50
"""

SINGLE_IMAGE = """
.. image:: images/%s
   :align: center
"""


def extract_docstring(filename):
    """ Extract a module-level docstring, if any
    """
    with open(filename) as f:
        lines = f.readlines()
    start_row = 0
    if lines[0].startswith('#!'):
        lines.pop(0)
        start_row = 1

    docstring = ''
    first_par = ''
    tokens = tokenize.generate_tokens(lines.__iter__)
    #~ for tok in tokens:
        #~ print(tok)
    with tokenize.open(filename) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for tok_type, tok_content, _, (erow, _), _ in tokens:
            tok_type = token.tok_name[tok_type]
            if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
                continue
            elif tok_type == 'STRING':
                docstring = eval(tok_content)
                # If the docstring is formatted with several paragraphs, extract
                # the first one:
                paragraphs = '\n'.join(line.rstrip()
                                       for line in docstring.split('\n')).split('\n\n')
                if len(paragraphs) > 0:
                    first_par = paragraphs[0]
            break
        return docstring, first_par, erow + 1 + start_row
            
    #~ for tok_type, tok_content, _, (erow, _), _ in tokens:
        #~ tok_type = token.tok_name[tok_type]
        #~ if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
            #~ continue
        #~ elif tok_type == 'STRING':
            #~ docstring = eval(tok_content)
            #~ # If the docstring is formatted with several paragraphs, extract
            #~ # the first one:
            #~ paragraphs = '\n'.join(line.rstrip()
                                   #~ for line in docstring.split('\n')).split('\n\n')
            #~ if len(paragraphs) > 0:
                #~ first_par = paragraphs[0]
        #~ break
    #~ return docstring, first_par, erow + 1 + start_row


def generate_all_example_rst(app):
    """ Generate the list of examples, as well as the contents of
        examples.
    """
    input_dir = os.path.abspath(app.builder.srcdir)
    example_dir = os.path.join(input_dir, 'examples')
    print ('*** Looking for examples in %s' % example_dir)
    generate_example_rst(example_dir,
                         os.path.join(input_dir, 'auto_examples'))

def generate_example_rst(example_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # we create an index.rst with all examples
    fhindex = open(os.path.join(out_dir, 'index.rst'), 'w')
    fhindex.write("""\

.. raw:: html

    <style type="text/css">
    .figure {
        float: left;
        margin: 10px;
        width: auto;
        height: 200px;
        width: 180px;
    }

    .figure img {
        display: inline;
        }

    .figure .caption {
        width: 170px;
        text-align: center !important;
    }
    </style>

Examples
========

.. _examples-index:
""")
    print('***** will generate_dir_rst', fhindex.name, example_dir, out_dir)
    generate_dir_rst('.', fhindex, example_dir, out_dir, False)
    fhindex.flush()


def generate_dir_rst(dir, fhindex, example_dir, out_dir, plot_anim):
    """ Generate the rst file for an example directory.
    """
    print('***')
    if not dir == '.':
        target_dir = os.path.join(out_dir, dir)
        src_dir = os.path.join(example_dir, dir)
    else:
        target_dir = out_dir
        src_dir = example_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for fname in sorted(os.listdir(src_dir)):
        print('***** found file', fname)
        if fname.endswith('data'):
            continue
        if fname.endswith('Notebooks'):
            continue
        if os.path.isdir(os.path.join(src_dir, fname)):
            print('***** this is a directory, going in...')
            # recursively treat this folder
            fhindex.write('\n' + fname + '\n')
            fhindex.write(len(fname) * '-' + '\n\n')
            if fname.endswith('animation'):
                generate_dir_rst(fname, fhindex, example_dir, out_dir, True)
            else:
                generate_dir_rst(fname, fhindex, example_dir, out_dir, False)
        if fname.endswith('.py'):
            print('generate_file_rst, plot_anim=%g' % plot_anim)
            generate_file_rst(fname, target_dir, src_dir, plot_anim)
            thumb = os.path.join(dir, 'images', 'thumb', fname[:-3] + '.png')
            link_name = fname
            fhindex.write('.. figure:: %s\n' % thumb)
            if link_name.startswith('._'):
                link_name = link_name[2:]
            if dir != '.':
                fhindex.write('   :target: ./%s/%s.html\n\n' % (dir, fname[:-3]))
            else:
                fhindex.write('   :target: ./%s.html\n\n' % link_name[:-3])
            fhindex.write('   %s\n\n' % link_name)
    fhindex.write("""
.. raw:: html

    <div style="clear: both"></div>
    """)  # clear at the end of the section


def generate_file_rst(fname, target_dir, src_dir, plot_anim):
    """ Generate the rst file for a given example.
    """
    base_image_name = os.path.splitext(fname)[0]
    image_fname = '%s.png' % base_image_name

    this_template = rst_template
    last_dir = os.path.split(src_dir)[-1]
    # to avoid leading . in file names, and wrong names in links
    if last_dir == '.' or last_dir == 'examples':
        last_dir = ''
    else:
        last_dir += '_'
    short_fname = last_dir + fname
    src_file = os.path.join(src_dir, fname)
    example_file = os.path.join(target_dir, fname)
    shutil.copyfile(src_file, example_file)

    image_dir = os.path.join(target_dir, 'images')
    thumb_dir = os.path.join(image_dir, 'thumb')
    print('image_dir: ', image_dir)
    print('thumb_dir: ', thumb_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)
    image_path = os.path.join(image_dir, image_fname)
    image_thumb_path = os.path.join(src_dir, 'thumb_' + image_fname)
    print('image_thumb_path: ', image_thumb_path)
    # copy image to the images folder
    shutil.copy(os.path.join(src_dir, image_fname), image_path)

    this_template = plot_rst_template
    # code moved to the actual examples not to depend on matplotlib
    # generate thumb file
    # thumb_file = os.path.join(thumb_dir, fname[:-3] + '.png')
    # from matplotlib import image
    # if os.path.exists(image_path):
    #    image.thumbnail(image_path, thumb_file, 0.2)

    thumb_file = os.path.join(thumb_dir, fname[:-3] + '.png')
    if not os.path.exists(image_thumb_path):
        # create something to replace the thumbnail
        shutil.copy('blank_image.png', thumb_file)
    else:
        shutil.copy(image_thumb_path, thumb_file)

    docstring, short_desc, end_row = extract_docstring(example_file)
    #docstring, short_desc, end_row = '', '', 0
    if plot_anim:
        image_fname = image_fname[:-4] + '.gif'
        gif_path = os.path.join(image_dir, image_fname)
        print('copying gif file to %s' % gif_path)
        # also copy animation file
        shutil.copy(os.path.join(src_dir, image_fname), gif_path)

    image_list = SINGLE_IMAGE % image_fname.lstrip('/')
    print('image_list is currently: %s' % image_list)

    f = open(os.path.join(target_dir, fname[:-2] + 'rst'), 'w')
    f.write(this_template % locals())
    f.flush()


def setup(app):
    app.connect('builder-inited', generate_all_example_rst)
