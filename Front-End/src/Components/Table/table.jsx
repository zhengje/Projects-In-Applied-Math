import './table.css'

const Table = ({KNN, SIM, IMF, GNN, JAC}) => {

    // Make Headers
    const headers = [
        {
            id: 1,
            KEY: "RANK",
            LABEL: "Rank",
        },
        {
            id: 2,
            KEY: "KNN",
            LABEL: "KNN Recs",
        },
        {
            id: 3,
            KEY: "SIM",
            LABEL: "SIM Recs",
        },
        {
            id: 4,
            KEY: "IMF",
            LABEL: "IMF Recs",
        },
        {
            id: 5,
            KEY: "GNN",
            LABEL: "GNN Recs",
        },
        {
            id: 6,
            KEY: "JAC",
            LABEL: "JAC Recs",
        },
        
    ]

    // Set Backend Data
    const data = []
    for (let i=0; i<10; i++){
        const temp = {
            ID: i+1,
            RANK: String(i+1),
            KNN: KNN[i]['title'],
            SIM: SIM[i]['title'],
            IMF: IMF[i]['title'],
            GNN: GNN[i]['title'],
            JAC: JAC[i]['title']
        }

        console.log(temp)
        data.push(temp)
    }

    // console.log(data)

    // temp data to test table formatting
    // const data = [
    //     {
    //         ID: 1,
    //         RANK: "1",
    //         KNN: "Slime Rancher",
    //         SIM: "Omori",
    //         IMF: "Dead by Daylight",
    //         GNN: "GTA 6",
    //         JAC: "YTTD: Death by Majority"
    //     },
    //     {
    //         ID: 2,
    //         RANK: "2",
    //         KNN: "Danganronpa: Trigger Happy Havoc",
    //         SIM: "Detroit Become Human",
    //         IMF: "IDV",
    //         GNN: "Feather Land",
    //         JAC: "Papa's Mocharia"
    //     },
    // ];


    return (
        <div>
        <table>
            {/* Map Headers */}
            <thead>
                <tr>
                    {headers.map((header, index) => (
                        <th key={index}>
                            <span>{header.LABEL}</span>
                        </th>
                    ))}
                </tr>
            </thead>

            {/* Map Data */}
            <tbody>
                {data.map((row, index) => (
                    <tr key={index}>
                        {headers.map((header, index) => {
                            return (
                                <td key={index}>
                                    {row[header.KEY]}
                                </td>
                            )
                        })}
                    </tr>
                ))}
            </tbody>
        </table>
        </div>
    )
}

export default Table