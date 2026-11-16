// Shows only rank and title for all games/methods
export default function Information({ rank, title}) {
    return (
        <li>
            RANK {rank}. {title}
        </li>
    )
}



// Shows rank, title, rating, and price (only rank and title for IMF)

// export default function Information({rank, title, price, rating}) {
// let content;

// if (price === 0) {
//     content = <li> RANK {rank}. {title} | {rating} | Free </li>
// } else if (!price) {
//     content = <li> RANK {rank}. {title} </li>
// } else {
//     content = <li> RANK {rank}. {title} | {rating} | ${price}</li>
// }

// return(
//    <div> {content} </div>
// )};




// First iteration

// export default function Information({ rank, title, price, rating}) {
//     return (
//     <div id="shrink-font">
//         { price > 0 ? (
//         <li>
//                 RANK {rank}. {title} | {rating} | ${price}
//         </li>
//         ) : (
//         <li>
//                 RANK {rank}. {title} | {rating} | Free
//         </li>
//         )
//         }

//     </div>
//     )
// }
