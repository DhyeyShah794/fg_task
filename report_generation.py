import os
import pandas as pd
import numpy as np

assignment_data = pd.read_csv("assignment_data.csv")
clustered_folder = "merged_clusters"
clusters = sorted(os.listdir(clustered_folder))

if "face_-1" in clusters:
    clusters.remove("face_-1")

performance_dict = {}

for cluster in clusters:
    cluster_files = os.listdir(os.path.join(clustered_folder, cluster))
    video_ids = set([file.split('_')[1] for file in cluster_files])

    performance_scores = []
    # Get the performance score for each video
    for video_id in video_ids:
        row_index = int(video_id)
        if row_index >= 0 and row_index < len(assignment_data):
            performance_score = assignment_data.iloc[row_index]['Performance']
            performance_scores.append(performance_score)
        mean_score = np.mean(performance_scores)
        std_dev = np.std(performance_scores)

        video_urls = [assignment_data.iloc[int(row_index)]['Video URL'] for row_index in video_ids]

        performance_dict[cluster] = {
            "average_performance": mean_score,
            "std_dev": std_dev,
            "video_urls": video_urls
        }

html_report_path = "influencer_report.html"

# Sort clusters in descending order of performance
sorted_performance = sorted(performance_dict.items(), key=lambda x: x[1]['average_performance'], reverse=True)


# Start the HTML document
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Influencer Performance Report</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 10px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
            cursor: pointer;
        }
        img {
            width: 100px;
            height: auto;
        }
    </style>
    <script>
        function sortTable(n) {
            const table = document.getElementById("clusterTable");
            let switching = true;
            let shouldSwitch;
            let dir = "asc"; 
            let switchcount = 0;
            while (switching) {
                switching = false;
                const rows = table.rows;
                for (let i = 1; i < rows.length - 1; i++) {
                    shouldSwitch = false;
                    let x = rows[i].getElementsByTagName("TD")[n];
                    let y = rows[i + 1].getElementsByTagName("TD")[n];
                    if (dir === "asc") {
                        if (parseFloat(x.innerHTML) > parseFloat(y.innerHTML)) {
                            shouldSwitch = true;
                            break;
                        }
                    } else if (dir === "desc") {
                        if (parseFloat(x.innerHTML) < parseFloat(y.innerHTML)) {
                            shouldSwitch = true;
                            break;
                        }
                    }
                }
                if (shouldSwitch) {
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount++;
                } else {
                    if (switchcount === 0 && dir === "asc") {
                        dir = "desc";
                        switching = true;
                    }
                }
            }
        }
    </script>
</head>
<body>
    <h1>Influencer Performance Report</h1>
    <table id="clusterTable">
        <thead>
            <tr>
                <th>ID</th>
                <th>Influencer Face</th>
                <th onclick="sortTable(2)">Average Performance Score</th>
                <th onclick="sortTable(3)">Standard Deviation</th>
                <th>Video URLs</th>
            </tr>
        </thead>
        <tbody>
"""

# Generate rows for the table
id_counter = 1
for cluster, value in sorted_performance:
    cluster_files = os.listdir(os.path.join(clustered_folder, cluster))
    if cluster_files:
        first_image = cluster_files[0]  # Use the first image from the cluster
        first_image_path = os.path.join(clustered_folder, cluster, first_image)
        # Add a row for this cluster
        html_content += f"""
        <tr>
            <td>{id_counter}</td>
            <td><img src="{first_image_path}" alt="Influencer Face"></td>
            <td>{value['average_performance'].round(4)}</td>
            <td>{value['std_dev'].round(4)}</td>
            <td>
                <ul>
        """
        for video_url in value['video_urls']:
            html_content += f"<li><a href='{video_url}'>{video_url}</a></li>"
        html_content += """
                </ul>
            </td>
        </tr>
        """
        id_counter += 1

html_content += """
        </tbody>
    </table>
</body>
</html>
"""

with open(html_report_path, "w") as html_file:
    html_file.write(html_content)

print(f"HTML report generated at: {html_report_path}")
