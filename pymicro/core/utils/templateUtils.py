#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for external script file templates manipulation

"""

## Imports
import os
import shutil
from subprocess import run
from pathlib import Path
from string import Template


# Script Template class
class ScriptTemplate():
    """Generic class to handle, fill and run script templates.
    """

    def __init__(self, template_file, script_file=None, args=None,
                 script_command=None, script_opts=None, workdir=None,
                 autodelete=False):
        """ScriptTemplate constructor.

        :param str template_file: Path to the script template file to handle
        :param dict args: Dictionnary of arguments values to fill in
            the script template file.
        :param str script_command: shell command to launch script
        :param list(str) script_command: list of shell command line options
            to add to `script_command` to launch script
        :param str workdir: path of the directory in which to run the script
        :param bool autodelete: If `True`, automatically removes the script
            file after running it, and when deleting the class instance.
        """
        self.file = Path(template_file).absolute()
        if args is not None:
            self.args = args
        else:
            args = {}
        self.set_script_filename(script_file)
        self.set_script_command(script_command)
        self.set_script_command_options(script_opts)
        self.set_work_directory(workdir)
        self.autodelete = False
        return

    def set_arguments(self, args=None, **kwargs):
        """Update script arguments value to fill in template.

        Arguments are updated in 2 steps:
            1. from the dict. `args` with all arguments:value pairs defined
               in the script template (if not `None`)
            2. from the individual arguments:value pairs passed as keywords
               args.

        :param dict args: Dictionnary of arguments values to fill in
            the script template file.
        **kwargs : any argument:value pair that can be filled in the script
            template
        """
        if args is not None:
            self.args.update(args)
        self.args.update(kwargs)
        return

    def set_script_filename(self, filename=None):
        """Set the filename to use to write script file"""
        if (filename is None) and not hasattr(self, 'script_file)'):
            self.script_file = ( self.file.parent / f"{self.file.stem}_tmp"
                            / self.file.suffix)
        else:
            self.script_file = Path(filename).absolute()
        return

    def set_work_directory(self, workdir=None):
        """Set the filename to use to write script file"""
        if (workdir is None) and not hasattr(self, 'workdir)'):
            self.workdir = os.getcwd()
        else:
            self.workdir = Path(workdir).absolute()
        return

    def set_script_command(self, shell_command=None):
        """Set the shell command to use to launch script.

        The shell command must be the executable file name of the software
        that is required to run the script file handled by the class instance,
        i.e. the first word of the shell command line used to run script.

        :param str script_command: shell command to launch script
            (examples: `matlab`, `Zset`, `gmsh`, `python`....)
        """
        self.command = shell_command
        return

    def set_script_command_options(self, command_opts=None):
        """Set the shell command to use to launch script with dedicated soft.

        The command options are the command line arguments that are given
        right after the shell executable name. In the following example of
        shell command line to run the script, the command_opts string is
        `-opts1 arg1 -opt2 arg21 arg22`

        .. code-block:: none
            executable_name -opts1 arg1 -opt2 arg21 arg22 ....

        :param list(str) script_command: list of shell command line options
            to add to `script_command` to launch script
        """
        self.opts = command_opts
        return

    def createScript(self, filename=None):
        """Create a script file from the template and the stored arguments.

        :param script_filename: Name of the script file to create. If not
            provided, uses ['self.file'+'_tmp'+extension].
        :type script_filename: TYPE, optional
        """
        self.set_script_filename(filename)
        # Create script file directory if needed
        if not self.script_file.parent.exists():
            os.mkdir(self.script_file.parent)
        # Get the template script file content
        with open(self.file, "r") as fin:
            temp = Template(fin.read())
        # Write the script file with the current input arguments values
        with open(self.script_file, "w") as f:
            f.write(temp.substitute(self.args))
        return

    def launchScript(self, workdir=None):
        """Run script file with provided shell command line and options.

        :param str workdir: directory in which to run the script. If None is
            passed, the used directory is the current work directory.
        :return output: output of the subprocess.run method used to launch the
            script.
        """
        if (self.command is None) or (self.opts is None):
            raise ValueError('None shell command or command options provided,'
                             ' cannot run the script.')
        if workdir is not None:self.set_work_directory(workdir)
        output = run(args=[self.command, *self.opts], shell=True,
                   capture_output=True, cwd=workdir)
        if self.autodelete:
            os.remove(self.script_file)
        return output


