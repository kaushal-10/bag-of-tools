import mido
outport = mido.open_output('SAMSUNG_Android MIDI 1')
outport.send(mido.Message('note_on', note=59, velocity=100))