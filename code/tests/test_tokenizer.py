import pytest
from lumbre.tokenizer import ByteBPE,replace_pair

@pytest.mark.parametrize('text',['','aaaaaaa','el gato mira el gato','á, ñ, \u4e2d\u6587 y un símbolo: \U0001f408'])
def test_bpe_roundtrip(text):
    t=ByteBPE.fit(text,25)
    assert t.decode(t.encode(text))==text
    assert ByteBPE.from_state(t.state())==t
    unseen='otra cadena: \u03bb'
    assert t.decode(t.encode(unseen))==unseen

def test_nonoverlap():
    assert replace_pair([1,1,1],(1,1),256)==[256,1]

def test_bpe_validation():
    with pytest.raises(ValueError): ByteBPE(((256,0),))
    with pytest.raises(ValueError): ByteBPE().decode([256])
    with pytest.raises(ValueError): ByteBPE.fit('x',-1)
