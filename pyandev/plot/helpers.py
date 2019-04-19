from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages



def check_pdf_arg(pdf):
    '''Function to check the type of the input and return an object of type PdfPages if a
    str is passed.
    
    Parameters
    ----------
    pdf : None or str or PdfPages
        Arg to check. If a str is passed then checks are performed that the parent directory
        exists and the filen is a .pdf file. If these checks pass then a PdfPages() object
        using the passed pdf arg is returned. If None is passed, None is returned. If a 
        PdfPages object is passed, it is simply returned back.

    Returns
    -------
    pdf : None or PdfPages
        Object of PdfPages type that can be used to add figures to. Or None if None is passed.

    '''

    if pdf is None:

        return pdf

    elif isinstance(pdf, str):

        pdf_path = Path(pdf)

        if not pdf_path.parent.exists():

            raise FileNotFoundError('pdf parent directory does not exist; ' + str(pdf_path.parent))

        if not pdf_path.suffix == '.pdf':

            raise TypeError('pdf not pdf file type; ' + pdf_path.suffix)

        return pdf

    elif isinstance(pdf, PdfPages):

        return pdf

    else:

        raise TypeError('unexpected type for pdf; ' + str(type(pdf)))



