# iMOACO<sub>&#8477;</sub> - PyTorch Implementation
=======
A PyTorch implementation of Falcón-Cardona and Coello's iMOACO<sub>&#8477;</sub>, an indicator-based many-objective ant colony optimizer for continuous search spaces.
The original journal article is available for purchase <a href="https://link.springer.com/chapter/10.1007%2F978-3-319-45823-6_36">here</a>,
while an older version is available to view <a href="http://computacion.cs.cinvestav.mx/%7ejfalcon/iMOACOR/iMOACOR-PPSN2016.pdf">here</a>.

During the creation of this implementation, a strong emphasis was placed on obtaining similar runtime performance as the original C implementation, which has been achieved even when using 1 CPU core.

Dependencies
------
<ul>
<li><b>Python3.5+</b></li>
<li><b>Numpy</b></li>
<li><b><a href="https://pytorch.org/get-started/locally/">PyTorch v1.0+</a></b>: Due to the current limitations of the JIT-compiler's tracing approach, techniques were implemented that are not compatible with the previous versions of PyTorch.</li>
</ul>

Usage
------
Set which configuration file will be used by editing the appropriate variables in <u>config.py</u>,
then run with <u>iMOACOR.py</u>.
For each run, the resulting Pareto optimal set and Pareto optimal front will be written to a subfolder within the <u>output</u> folder.