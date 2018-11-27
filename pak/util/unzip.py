import zipfile
import tarfile
import lzma
import subprocess
from pak import utils
import time
import os
from os.path import isfile


def has_os_unzip():
    """ Checks if an OS tool for unzipping is installed

        returns True if so, o/w False
    """
    command = ["which", 'unzip']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, err = process.communicate()
    return len(output) > 1  # path to binary OR empty


def has_os_unxz():
    """ Checks if an OS tool for unxz is installed (needed for .xz files)

        returns True if so, o/w False
    """
    command = ["which", 'unxz']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, err = process.communicate()
    return len(output) > 1  # path to binary OR empty


def unzip_using_os_tools(fzip, export_folder, verbose):
    """ unzips the given file to the given folder using os tools
    """
    #TODO: this has to be figured out eventually
    command = ["unzip", fzip, '-d', export_folder]
    utils.talk('\ttry unzip using OS tools', verbose)
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    utils.talk('\tunzip using OS tools', verbose)
    output, err = process.communicate()
    utils.talk('\terr:\t' + str(err), verbose)


def unzip(fzip, export_folder, verbose=False, force_os_tools=False,
          del_after_unzip=False):
    """
        @param fzip: {String} full path to the zip-file
        @param export_folder: {String} place to unzip
        @param verbose: {boolean} if true: debug
        @param force_os_tools: {boolean} if true: try to use OS tools
                                This is a horrible hack but it seems that
                                the python tools cannot unzip certain zip files
                                (that seem to be zipped on OSX) and I have no
                                time to find a 'good' solution so the ugly hack
                                is: use the system tools which for some reason
                                work! (on LINUX!!)
        @param del_after_unzip: delete after unzipping
    """
    if force_os_tools:
        assert has_os_unzip()
        unzip_using_os_tools(fzip, export_folder, verbose)
    else:
        if fzip.endswith('.zip'):
            try:
                utils.talk("unzip " + fzip + " -> " + export_folder, verbose)
                zip_ref = zipfile.ZipFile(fzip, 'r')
                zip_ref.extractall(export_folder)
                zip_ref.close()
            except zipfile.BadZipFile as e:
                utils.talk('unzip exception:' + str(e), verbose)

                # -- try it the hard way -- use OS tools
                if has_os_unzip():
                    unzip_using_os_tools(fzip, export_folder, verbose)
                else:
                    raise

        elif fzip.endswith('tar.gz'):
            mode = 'r:gz'
            utils.talk("untar " + fzip + " -> " + export_folder, verbose)
            tar = tarfile.open(fzip, mode)
            tar.extractall(export_folder)
            tar.close()
        elif fzip.endswith('tar.bz2'):
            command = ['tar', 'xvjf', fzip]
            utils.talk('\ttry to untar using OS tools', verbose)
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
            utils.talk('\tuntar using OS tools', verbose)
            output, err = process.communicate()
            process.wait()
            utils.talk('\tuntar finished', verbose)
            utils.talk('\t\terrors:\t' + str(err), verbose)
        elif fzip.endswith('.xz'):
            command = ["unxz", fzip]
            utils.talk('\ttry unxz using OS tools', verbose)
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
            utils.talk('\tunxz using OS tools', verbose)
            output, err = process.communicate()
            process.wait()
            utils.talk('\terr:\t' + str(err), verbose)
        else:
            raise RuntimeError("No unzip routine found for " + fzip)

        if del_after_unzip:
            time.sleep(0.5)  # just to be sure sleep some time
            if isfile(fzip):
                os.remove(fzip)