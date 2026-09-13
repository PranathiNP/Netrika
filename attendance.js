(function () {
    const container = document.getElementById('attendanceReports');
    const status = document.getElementById('attendanceStatus');
    const analyticsStatus = document.getElementById('analyticsStatus');
    const teacher = document.body.classList.contains('teacher-dashboard');
    document.querySelectorAll('[data-panel]').forEach(function (button) {
        button.addEventListener('click', function () {
            document.querySelectorAll('[data-dashboard-panel]').forEach(function (panel) {
                panel.hidden = panel.id !== button.dataset.panel;
            });
            document.getElementById('profileMenu').hidden = true;
            document.getElementById('profileButton').setAttribute('aria-expanded', 'false');
        });
    });
    document.addEventListener('click', function (event) {
        if (!event.target.closest('.profile-area')) {
            document.getElementById('profileMenu').hidden = true;
            document.getElementById('profileButton').setAttribute('aria-expanded', 'false');
        }
    });
    function renderAnalytics(reports) {
        const latest = reports[0];
        const total = reports.reduce(function (sum, report) { return sum + report.students.length; }, 0);
        const present = reports.reduce(function (sum, report) { return sum + report.present_count; }, 0);
        document.getElementById('sessionMetric').textContent = reports.length;
        document.getElementById('presentMetric').textContent = teacher ? (latest ? latest.present_count : 0) : present;
        document.getElementById('presentLabel').textContent = teacher ? 'Present in latest session' : 'Sessions attended';
        document.getElementById('rateMetric').textContent = (total ? Math.round(present / total * 100) : 0) + '%';
        const trend = document.getElementById('attendanceTrend');
        trend.replaceChildren();
        reports.slice(0, 8).reverse().forEach(function (report) {
            const rate = report.students.length ? Math.round(report.present_count / report.students.length * 100) : 0;
            const row = document.createElement('div');
            const label = document.createElement('span');
            label.textContent = new Date(report.created_at).toLocaleString();
            const bar = document.createElement('progress');
            bar.max = 100;
            bar.value = rate;
            bar.setAttribute('aria-label', label.textContent + ': ' + rate + '% present');
            const value = document.createElement('strong');
            value.textContent = rate + '%';
            row.append(label, bar, value);
            trend.appendChild(row);
        });
        analyticsStatus.textContent = reports.length ? 'Updated ' + new Date().toLocaleTimeString() : 'No attendance sessions yet.';
    }
    let loading = false;
    window.loadAttendance = async function () {
        if (loading) return;
        loading = true;
        status.textContent = 'Loading attendance…';
        try {
            const response = await fetch('/api/attendance');
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || 'Could not load attendance.');
            renderAnalytics(data.reports);
            container.replaceChildren();
            data.reports.forEach(function (report) {
                const section = document.createElement('section');
                const title = document.createElement('h3');
                title.textContent = new Date(report.created_at).toLocaleString() + ' — ' + report.source_name;
                section.appendChild(title);
                const summary = document.createElement('p');
                summary.textContent = report.present_count + ' / ' + report.students.length + ' present';
                if (report.unknown_detections) summary.textContent += ' · ' + report.unknown_detections + ' unknown face detections (may repeat across video frames)';
                section.appendChild(summary);
                const table = document.createElement('table');
                const head = table.createTHead().insertRow();
                ['Student', 'Attendance'].forEach(function (label) {
                    const cell = document.createElement('th');
                    cell.scope = 'col';
                    cell.textContent = label;
                    head.appendChild(cell);
                });
                const body = table.createTBody();
                report.students.forEach(function (student) {
                    const row = body.insertRow();
                    row.insertCell().textContent = student.name;
                    row.insertCell().textContent = student.status;
                });
                section.appendChild(table);
                container.appendChild(section);
            });
            status.textContent = data.reports.length ? '' : 'No attendance sessions yet.';
        } catch (error) {
            status.textContent = error.message;
            analyticsStatus.textContent = error.message + ' Retrying automatically.';
        } finally {
            loading = false;
        }
    };
    document.getElementById('refreshAttendance').addEventListener('click', window.loadAttendance);
    window.loadAttendance();
    window.setInterval(function () { if (!document.hidden) window.loadAttendance(); }, 15000);
    document.addEventListener('visibilitychange', function () { if (!document.hidden) window.loadAttendance(); });
    const upload = document.getElementById('uploadEnrollment');
    if (!upload) return;
    const enrollmentStatus = document.getElementById('enrollmentStatus');
    fetch('/api/face-registration').then(function (response) {
        return response.json();
    }).then(function (data) {
        enrollmentStatus.textContent = data.message || (data.saved_frames + ' face images saved.');
    }).catch(function () { enrollmentStatus.textContent = 'Could not load enrollment status.'; });
    upload.addEventListener('click', async function () {
        const input = document.getElementById('enrollmentVideo');
        const file = input.files[0];
        if (!file || !file.type.startsWith('video/') || file.size > 30000000) {
            enrollmentStatus.textContent = 'Select a video smaller than 30 MB.';
            return;
        }
        upload.disabled = true;
        input.disabled = true;
        enrollmentStatus.textContent = 'Extracting and saving face images…';
        try {
            const body = new FormData();
            body.append('media', file);
            const response = await fetch('/api/face-registration/video', {method: 'POST', body: body});
            const data = await response.json();
            enrollmentStatus.textContent = data.message;
        } catch (error) {
            enrollmentStatus.textContent = 'Could not upload the video. Please try again.';
        } finally {
            upload.disabled = false;
            input.disabled = false;
        }
    });
})();
