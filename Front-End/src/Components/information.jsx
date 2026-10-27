export default function Information({ rank, title, price, rating}) {
    return (
    <div id="shrink-font">
        { price > 0 ? (
        <li>
                RANK {rank}. {title} | {rating} | ${price}
        </li>
        ) : (
        <li>
                RANK {rank}. {title} | {rating} | Free
        </li>
        )
        }

    </div>
    )
}