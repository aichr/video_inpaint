import os
import platform


def build_ebsynth():
    if os.path.exists('ebsynth/bin/ebsynth'):
        print('Ebsynth has been built.')
        return

    os_str = platform.system()

    if os_str == 'Windows':
        print('Build Ebsynth Windows 64 bit.',
              'If you want to build for 32 bit, please modify install.py.')
        cmd = '.\\build-win64-cpu+cuda.bat'
        exe_file = 'ebsynth/bin/ebsynth.exe'
    elif os_str == 'Linux':
        cmd = 'bash build-linux-cpu+cuda.sh'
        exe_file = 'ebsynth/bin/ebsynth'
    elif os_str == 'Darwin':
        cmd = 'sh build-macos-cpu_only.sh'
        exe_file = 'ebsynth/bin/ebsynth.app'
    else:
        print('Cannot recognize OS. Ebsynth installation stopped.')
        return

    os.chdir('ebsynth')
    print(cmd)
    os.system(cmd)
    os.chdir('../..')
    if os.path.exists(exe_file):
        print('Ebsynth installed successfully.')
    else:
        print('Failed to install Ebsynth.')


if __name__ == '__main__':
    build_ebsynth()
