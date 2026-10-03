fn main() {
    // Emits the link flags for librtlsdr; fail early with an install hint if it is missing.
    if let Err(e) = pkg_config::Config::new().probe("librtlsdr") {
        panic!(
            "\n\nlibrtlsdr development files not found ({e}).\n\
             Install them and rebuild:\n  \
             Fedora:        sudo dnf install rtl-sdr-devel\n  \
             Debian/Ubuntu: sudo apt install librtlsdr-dev\n\n"
        );
    }
}
