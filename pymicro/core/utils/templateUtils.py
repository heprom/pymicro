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
    
    # TODO: __repr__ method for string representation
    # TODO: clean files method --> deletes temp file

    def __init__(self, template_file, script_file=None, args=None,
                 script_command=None, script_opts=[], workdir=None,
                 autodelete=False, verbose=False):
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
            self.args = {}
        self.set_script_filename(script_file)
        self.set_script_command(script_command)
        self.set_script_command_options(script_opts)
        self.set_work_directory(workdir)
        self.autodelete = autodelete
        self._verbose = verbose
        return
    
    def __del__(self):
        """ScriptTemplate class destructor."""
        if self.autodelete:
            self.clean_script_file()
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
    
    def set_template_filename(self, filename=None):
        """Set the filename to use to write script file"""
        if filename is not None:
            self.file = Path(filename).absolute()
        return

    def set_script_filename(self, filename=None):
        """Set the filename to use to write script file"""
        if (filename is None) and not hasattr(self, 'script_file)'):
            self.script_file = ( self.file.parent / f"{self.file.stem}_tmp"
                                f"{self.file.suffix}")
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
        if command_opts is None:
            self.opts = []
        else:
            self.opts = command_opts
        return
    
    def clean_script_file(self):
        """Remove the last written script file."""
        if self.script_file.exists():
            print('Removing {}'.format(self.script_file))
            os.remove(self.script_file)
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
        return self.script_file

    def runScript(self, workdir=None, append_filename=True,
                  print_output=False):
        """Run script file with provided shell command line and options.

        :param str workdir: directory in which to run the script. If None is
            passed, the used directory is the current work directory.
        :param bool print_output: if `True` print the script return code,
            standard output and standard error.
        :param bool append_filename: If `True` (default), add the script
            filename at the end of the command line used to run the script
        :return output: output of the subprocess.run method used to launch the
            script.
        """
        if self.command is None:
            raise ValueError('None shell command provided, cannot run the'
                             ' script.')
        if workdir is not None: self.set_work_directory(workdir)
        # create script file
        self.createScript()
        # create shell command line to run script
        command_line = [self.command, *self.opts]
        if append_filename:
            command_line.append(self.script_file)
        if self._verbose:
            print(' ---- Launching Script ----')
            print(f'Work directory : {self.workdir}')
            print(f'Script file: {self.script_file}')
            print(f'Executable : {self.command}')
            print(f'Options : {self.opts}')
            print(f'Shell Command line: {command_line}')
        output = run(args=command_line, cwd=self.workdir,
                     capture_output=True)
        if self.autodelete:
            self.clean_script_file()
        if print_output:
            print(f"SCRIPT RETURN CODE IS {output.returncode}.\n")
            print(f"SCRIPT STDOUT is: \n\n{output.stdout.decode()}")
            print(f"SCRIPT STDERR is: \n\n{output.stderr.decode()}")
        return  output


