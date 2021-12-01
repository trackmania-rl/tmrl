Install
-------

Install through pypi using pip (or your favourite trendy tool)::

    pip install inputs

Or download it from github::

    git clone https://github.com/zeth/inputs.git
    cd inputs
    python setup.py install

Inputs written is in pure Python and there are no dependencies on
Raspberry Pi, Linux or Windows. On the Mac, inputs needs PyObjC which
the included setup.py file will install automatically (as will pip).

Windows permissions
-------------------

By default Windows doesn't stop inputs. However, if you have some
third-party security software you may need to white-list Python. Try
it and find out.

Linux permissions
-----------------

On the Raspberry Pi's Raspbian everything just works.

However, each Linux distribution is different. Some will work straight
away, for some you need to fiddle with permissions.

Linux distributions often (quite rightly) assume that applications are
installed through their package manager and given the relevant
permissions to access the input devices. However, inputs.py is brand
new and not yet packaged by any Linux distribution.

Therefore, if the inputs module works as root (e.g. using sudo) but
not as your normal user, then you usually need to add yourself to an
inputs group or similar.

Mac permissions
---------------

On the Mac, until you write a proper installer for your program, you
will probably have to use the settings application to allow your
program to access the input devices.

    .. image:: https://raw.githubusercontent.com/zeth/inputs/master/macsecurity.png

The first time you use inputs, it will not have any output, then you
will either get the above settings window pop up automatically, or you
will need to find your way there.
