import retro.data
import os
import sys

try:
    rom_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Super Mario Bros. (World).nes')
    rom_file = open(rom_file_path, "rb")
    data, hash = retro.data.groom_rom(rom_file_path, rom_file)

    known_hashes = retro.data.get_known_hashes()

    if hash in known_hashes:
        game, ext, curpath = known_hashes[hash]
        with open(os.path.join(curpath, game, 'rom%s' % ext), 'wb') as f:
            f.write(data)
except:
    print('failed to import ROM file', file=sys.stderr)
